"""
MonoDDLE 日志系统 - 基于 Rich + Loguru 的统一日志模块

提供美观的控制台输出和结构化的文件日志记录。
支持分布式训练环境下的日志管理。
"""
import os
import sys
from typing import Optional, Union
from functools import wraps

from loguru import logger
from rich.console import Console
from rich.theme import Theme
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

# 安装 Rich 的美化 traceback
install_rich_traceback(show_locals=True, width=120)

# 自定义主题
MONODLE_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "title": "bold magenta",
    "highlight": "bold blue",
    "metric": "green",
    "epoch": "bold cyan",
    "loss": "yellow",
})

# 全局 Console 实例
console = Console(theme=MONODLE_THEME, force_terminal=True)


def _log_rich_content(content):
    """
    Helper function to log rich content to file via loguru
    """
    if MonoDDLELogger._instance and MonoDDLELogger._instance.log_file and MonoDDLELogger._instance.is_main:
        try:
            from io import StringIO
            buf = StringIO()
            temp_console = Console(file=buf, force_terminal=False, width=160, color_system=None)
            temp_console.print(content)
            text = buf.getvalue().rstrip()
            if text:
                # 使用 bind(console=False) 防止输出到控制台，仅记录到文件
                logger.bind(console=False).info("\n" + text)
        except Exception:
            pass


def print_dict_table(data: dict, title: str = "Metrics"):
    """
    打印字典数据为表格
    """
    table = Table(title=title, border_style="bold magenta", show_header=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    for k, v in data.items():
        if isinstance(v, float):
            val_str = f"{v:.4f}"
        else:
            val_str = str(v)
        table.add_row(k, val_str)

    console.print(table)
    _log_rich_content(table)


def print_kitti_eval_results(rich_data, prev_rich_data=None):
    """
    使用 Rich 表格打印 KITTI 评估结果
    rich_data: list of dicts collected in get_official_eval_result
    prev_rich_data: list of dicts from previous evaluation (optional)
    """

    # Helper to find matching previous data
    def get_prev_block(p_data, c_name, o_str):
        if not p_data: return None
        for item in p_data:
            if item['class_name'] == c_name and item['overlap_str'] == o_str:
                return item
        return None

    def format_score_with_diff(score, prev_score):
        score_txt = f"{score:.4f}"
        if prev_score is None:
            return score_txt
        
        diff = score - prev_score
        if abs(diff) < 1e-4:
            return score_txt
        
        # Color the diff
        if diff > 0:
            diff_styled = f"[green]↑{abs(diff):.4f}[/]" 
        else:
            diff_styled = f"[red]↓{abs(diff):.4f}[/]"
            
        return f"{score_txt} {diff_styled}"

    for item in rich_data:
        class_name = item['class_name']
        overlap_str = item['overlap_str']
        
        prev_item = get_prev_block(prev_rich_data, class_name, overlap_str)
        
        # AP Table
        table_ap = Table(title=f"{class_name} AP@{overlap_str}", border_style="blue", box=None)
        table_ap.add_column("Metric", style="bold cyan")
        table_ap.add_column("Easy", justify="right", style="green")
        table_ap.add_column("Mod.", justify="right", style="yellow")
        table_ap.add_column("Hard", justify="right", style="red")

        for metric_name, scores in item['metrics'].items():
            prev_scores = prev_item['metrics'].get(metric_name) if prev_item and metric_name in prev_item['metrics'] else None
            
            row_data = [f"{metric_name} AP"]
            for i in range(3): # Easy, Mod, Hard
                s = scores[i]
                p = prev_scores[i] if prev_scores else None
                row_data.append(format_score_with_diff(s, p))
                
            table_ap.add_row(*row_data)
        
        console.print(table_ap)
        _log_rich_content(table_ap)
        
        # AP R40 Table
        table_r40 = Table(title=f"{class_name} AP_R40@{overlap_str}", border_style="magenta", box=None)
        table_r40.add_column("Metric", style="bold cyan")
        table_r40.add_column("Easy", justify="right", style="green")
        table_r40.add_column("Mod.", justify="right", style="yellow")
        table_r40.add_column("Hard", justify="right", style="red")

        for metric_name, scores in item['metrics_R40'].items():
            prev_scores = prev_item['metrics_R40'].get(metric_name) if prev_item and metric_name in prev_item['metrics_R40'] else None
            
            row_data = [f"{metric_name} AP"]
            for i in range(3): # Easy, Mod, Hard
                s = scores[i]
                p = prev_scores[i] if prev_scores else None
                row_data.append(format_score_with_diff(s, p))

            table_r40.add_row(*row_data)
        
        console.print(table_r40)
        _log_rich_content(table_r40)
        console.print("") # spacing


class MonoDDLELogger:
    """
    MonoDDLE 项目的统一日志管理器
    
    特性:
    - 基于 loguru 的强大日志功能
    - 基于 rich 的美观控制台输出
    - 支持分布式训练 (只在主进程输出)
    - 自动文件日志记录
    - 结构化日志格式
    """
    
    _instance: Optional['MonoDDLELogger'] = None
    _initialized: bool = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        rank: int = 0,
        level: str = "INFO",
        rotation: str = "100 MB",
        retention: str = "30 days",
    ):
        """
        初始化日志管理器
        
        Args:
            log_file: 日志文件路径 (可选)
            rank: 分布式训练中的进程 rank
            level: 日志级别
            rotation: 日志文件轮转大小
            retention: 日志文件保留时间
        """
        if self._initialized:
            return
            
        self.rank = rank
        self.is_main = rank == 0
        self.level = level if self.is_main else "ERROR"
        self.log_file = log_file
        
        # 移除默认的 handler
        logger.remove()
        
        # 自定义格式
        console_format = (
            "<level>{level.icon}</level> "
            "<cyan>{time:HH:mm:ss}</cyan> | "
            "<level>{message}</level>"
        )
        
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
        
        # 添加 level 图标
        logger.level("DEBUG", icon="🔍")
        logger.level("INFO", icon="ℹ️ ")
        logger.level("SUCCESS", icon="✅")
        logger.level("WARNING", icon="⚠️ ")
        logger.level("ERROR", icon="❌")
        logger.level("CRITICAL", icon="💀")
        
        # 自定义 sink 函数，确保输出正确换行并使用 Rich console
        def rich_sink(message):
            # message.record 包含日志记录的所有信息
            record = message.record
            level_icon = record["level"].icon
            time_str = record["time"].strftime("%H:%M:%S")
            msg = record["message"]
            level_name = record["level"].name.lower()
            
            # 根据级别设置颜色
            level_colors = {
                "debug": "dim",
                "info": "bold",
                "success": "bold green",
                "warning": "bold yellow",
                "error": "bold red",
                "critical": "bold red reverse",
            }
            color = level_colors.get(level_name, "")
            
            # 使用 Rich console 输出，自动处理换行
            console.print(f"[{color}]{level_icon}[/{color}] [cyan]{time_str}[/cyan] | [{color}]{msg}[/{color}]")
        
        # 控制台输出 (使用 Rich)
        if self.is_main:
            logger.add(
                rich_sink,
                format="{message}",  # 格式在 sink 中处理
                level=self.level,
                colorize=False,  # 由 Rich 处理颜色
                filter=lambda record: record["extra"].get("console", True)
            )
        
        # 文件输出
        if log_file and self.is_main:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            logger.add(
                log_file,
                format=file_format,
                level="DEBUG",  # 文件记录所有级别
                rotation=rotation,
                retention=retention,
                encoding="utf-8",
                enqueue=True,  # 线程安全
            )
        
        self._initialized = True
        
        if self.is_main:
            self._print_banner()
    
    def _print_banner(self):
        """打印启动 banner"""
        banner_text = Text()
        banner_text.append("MonoDDLE", style="bold magenta")
        banner_text.append(" - Monocular 3D Object Detection", style="cyan")
        
        panel = Panel(
            banner_text,
            title="[bold blue]🚗 MonoDDLE[/bold blue]",
            subtitle="[dim]Logging System Initialized[/dim]",
            border_style="blue",
        )
        console.print(panel)
        _log_rich_content(panel)
    
    @property
    def logger(self):
        """返回 loguru logger 实例"""
        return logger
    
    # ============ 日志方法 ============
    
    def debug(self, message: str, *args, **kwargs):
        """Debug 级别日志"""
        if self.is_main:
            logger.opt(depth=1).debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Info 级别日志"""
        if self.is_main:
            logger.opt(depth=1).info(message, *args, **kwargs)
    
    def success(self, message: str, *args, **kwargs):
        """Success 级别日志"""
        if self.is_main:
            logger.opt(depth=1).success(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Warning 级别日志"""
        if self.is_main:
            logger.opt(depth=1).warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Error 级别日志"""
        logger.opt(depth=1).error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Critical 级别日志"""
        logger.opt(depth=1).critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """记录异常信息"""
        logger.opt(depth=1).exception(message, *args, **kwargs)
    
    # ============ Rich 美化输出 ============
    
    def print_title(self, title: str, style: str = "title"):
        """打印标题"""
        if not self.is_main:
            return
        console.rule(f"[{style}]{title}[/{style}]", style=style)
        _log_rich_content(Rule(f"[{style}]{title}[/{style}]", style=style))
    
    def print_section(self, title: str, content: str = ""):
        """打印章节"""
        if not self.is_main:
            return
        msg = f"\n[bold blue]{'='*20}  {title}  {'='*20}[/bold blue]"
        console.print(msg)
        _log_rich_content(msg)
        if content:
            console.print(content)
            _log_rich_content(content)
    
    def print_config(self, config: dict, title: str = "Configuration"):
        """以表格形式打印配置"""
        if not self.is_main:
            return
        
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        def add_items(items, prefix=""):
            for key, value in items.items():
                if isinstance(value, dict):
                    add_items(value, f"{prefix}{key}.")
                else:
                    table.add_row(f"{prefix}{key}", str(value))
        
        add_items(config)
        console.print(table)
        _log_rich_content(table)
    
    def print_metrics(self, metrics: dict, title: str = "Metrics", highlight_key: Optional[str] = None):
        """以表格形式打印评估指标"""
        if not self.is_main:
            return
        
        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green", justify="right")
        
        for key, value in metrics.items():
            style = "bold green" if key == highlight_key else "green"
            if isinstance(value, float):
                table.add_row(key, f"[{style}]{value:.4f}[/{style}]")
            else:
                table.add_row(key, f"[{style}]{value}[/{style}]")
        
        console.print(table)
        _log_rich_content(table)
    
    def print_training_status(
        self,
        epoch: int,
        max_epoch: int,
        batch: int,
        total_batches: int,
        loss: float,
        lr: float,
        data_time: float = 0.0,
        iter_time: float = 0.0,
        stats_dict: dict = None,
    ):
        """打印训练状态"""
        if not self.is_main:
            return
        
        status = (
            f"[epoch]Epoch[/epoch] [{epoch}/{max_epoch}] | "
            f"[highlight]Iter[/highlight] [{batch}/{total_batches}] | "
            f"[loss]Loss[/loss]: {loss:.6f} | "
            f"[info]LR[/info]: {lr:.2e}"
        )
        if data_time > 0:
            status += f" | [dim]Data: {data_time:.3f}s[/dim]"
        if iter_time > 0:
            status += f" | [dim]Iter: {iter_time:.3f}s[/dim]"

        if stats_dict:
            # 格式化各个 loss 组件
            loss_components = []
            for k, v in stats_dict.items():
                loss_components.append(f"{k}: {v:.4f}")
            if loss_components:
                status += " | [cyan]" + " ".join(loss_components) + "[/cyan]"
        
        console.print(status)
        _log_rich_content(status)
    
    def print_checkpoint_info(self, checkpoint_path: str, action: str = "Saved"):
        """打印 checkpoint 信息"""
        if not self.is_main:
            return
        self.success(f"Checkpoint {action}: {checkpoint_path}")
    
    def print_model_summary(self, model, input_size: Optional[tuple] = None):
        """打印模型摘要"""
        if not self.is_main:
            return
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        table = Table(title="Model Summary", show_header=True, header_style="bold magenta")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green", justify="right")
        
        table.add_row("Total Parameters", f"{total_params:,}")
        table.add_row("Trainable Parameters", f"{trainable_params:,}")
        table.add_row("Non-trainable Parameters", f"{total_params - trainable_params:,}")
        table.add_row("Model Size (MB)", f"{total_params * 4 / 1024 / 1024:.2f}")
        
        console.print(table)
        _log_rich_content(table)
    
    # ============ 进度条工具 ============

def create_progress_bar(description: str = "Processing", transient: bool = False) -> Progress:
    """创建美观的进度条"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
    )


def create_epoch_progress(total_epochs: int, description: str = "Training") -> Progress:
    """创建 epoch 级别的进度条"""
    return Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[bold magenta]{task.description}"),
        BarColumn(bar_width=30, style="magenta", complete_style="green"),
        TaskProgressColumn(),
        TextColumn("[cyan]•[/cyan]"),
        TimeElapsedColumn(),
        TextColumn("[cyan]•[/cyan]"),
        TimeRemainingColumn(),
        console=console,
    )


# ============ 便捷导出 ============

__all__ = [
    'MonoDDLELogger',
    'console',
    'create_progress_bar',
    'create_epoch_progress',
]
