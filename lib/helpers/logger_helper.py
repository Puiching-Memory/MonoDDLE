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
from rich import box
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
    使用 Rich 表格打印 KITTI 评估结果 (Markdown 风格统一表格)
    rich_data: list of dicts collected in get_official_eval_result
    prev_rich_data: list of dicts from previous evaluation (optional)
    """

    # 按类名和 overlap 分组
    data_by_class = {}
    for item in rich_data:
        c_name = item['class_name']
        if c_name not in data_by_class:
            data_by_class[c_name] = {}
        data_by_class[c_name][item['overlap_str']] = item
        
    prev_by_class = {}
    if prev_rich_data:
        for item in prev_rich_data:
            c_name = item['class_name']
            if c_name not in prev_by_class:
                prev_by_class[c_name] = {}
            prev_by_class[c_name][item['overlap_str']] = item

    def get_score(item, m_type, m_name, diff_idx):
        if not item or m_type not in item:
            return None
        scores = item[m_type].get(m_name)
        if scores is None or diff_idx >= len(scores):
            return None
        return float(scores[diff_idx])

    def format_val(val, prev_val):
        if val is None:
            return "-"
        score_txt = f"{val:.2f}"
        if prev_val is None:
            return score_txt
        diff = val - prev_val
        if abs(diff) < 1e-4:
            return score_txt
        diff_styled = f"[green]↑{abs(diff):.2f}[/]" if diff > 0 else f"[red]↓{abs(diff):.2f}[/]"
        return f"{score_txt} {diff_styled}"

    for cat_name in ['Car', 'Pedestrian', 'Cyclist']:
        if cat_name not in data_by_class:
            continue

        overlaps = data_by_class[cat_name]
        prev_overlaps = prev_by_class.get(cat_name, {})

        # KITTI 评估通常关注 R40
        if cat_name == 'Car':
            primary_key = '0.70, 0.70, 0.70'
            secondary_key = '0.70, 0.50, 0.50'
        else:
            primary_key = '0.50, 0.50, 0.50'
            secondary_key = '0.50, 0.25, 0.25'

        item_p = overlaps.get(primary_key)
        item_s = overlaps.get(secondary_key)
        prev_p = prev_overlaps.get(primary_key)
        prev_s = prev_overlaps.get(secondary_key)

        if not item_p:
            continue

        logger.info(f"Official Evaluation Results for {cat_name}:")

        # 打印 R40 统一表格 (与 README KITTI 验证集风格一致)
        table = Table(
            title=f"{cat_name} AP_R40 Performance (Standard Format)",
            header_style="bold cyan",
            border_style="magenta",
            box=box.MARKDOWN,
        )
        table.add_column("Type", style="bold", no_wrap=True)
        
        # 定义要展示的 R40 列 (label, metric_name, item_source, prev_item_source)
        cols = [
            ("3D@0.7 (E)", "3d", item_p, prev_p),
            ("3D@0.7 (M)", "3d", item_p, prev_p),
            ("3D@0.7 (H)", "3d", item_p, prev_p),
            ("BEV@0.7 (E)", "bev", item_p, prev_p),
            ("BEV@0.7 (M)", "bev", item_p, prev_p),
            ("BEV@0.7 (H)", "bev", item_p, prev_p),
        ]
        
        if cat_name == 'Car':
            if item_s:
                cols.extend([
                    ("3D@0.5 (E)", "3d", item_s, prev_s),
                    ("3D@0.5 (M)", "3d", item_s, prev_s),
                    ("3D@0.5 (H)", "3d", item_s, prev_s),
                    ("BEV@0.5 (E)", "bev", item_s, prev_s),
                    ("BEV@0.5 (M)", "bev", item_s, prev_s),
                    ("BEV@0.5 (H)", "bev", item_s, prev_s),
                ])
            cols.extend([
                ("AOS (E)", "aos", item_p, prev_p),
                ("AOS (M)", "aos", item_p, prev_p),
                ("AOS (H)", "aos", item_p, prev_p),
            ])
        elif item_s:
            cols.extend([
                ("3D@Sec (E)", "3d", item_s, prev_s),
                ("3D@Sec (M)", "3d", item_s, prev_s),
                ("3D@Sec (H)", "3d", item_s, prev_s),
            ])

        for label, _, _, _ in cols:
            style = "green" if "(E)" in label else "yellow" if "(M)" in label else "red"
            table.add_column(label, justify="right", style=style)

        # 添加 R40 行
        r40_row = ["AP_R40"]
        # 我们需要知道每个指标对应的 idx (0:easy, 1:mod, 2:hard)
        # 这里借助 enumerate 和取模
        for i, (label, m_name, it, pit) in enumerate(cols):
            diff_idx = i % 3 # 虽然 cols 列表是平铺的，但每 3 个是一组 (E, M, H)
            val = get_score(it, 'metrics_R40', m_name, diff_idx)
            pval = get_score(pit, 'metrics_R40', m_name, diff_idx)
            r40_row.append(format_val(val, pval))
        
        table.add_row(*r40_row)
        
        # 可选：如果也想看 AP_11，可以加一行
        ap11_row = ["AP_11"]
        for i, (label, m_name, it, pit) in enumerate(cols):
            diff_idx = i % 3
            val = get_score(it, 'metrics', m_name, diff_idx)
            pval = get_score(pit, 'metrics', m_name, diff_idx)
            ap11_row.append(format_val(val, pval))
        table.add_row(*ap11_row)

        console.print(table)
        _log_rich_content(table)
        console.print("")


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


# ============ CSV 评估结果保存 ============

def save_eval_to_csv(rich_data, csv_path, model_name="unknown", epoch=None):
    """
    将评估结果追加保存到 CSV 文件。

    Parameters
    ----------
    rich_data : list[dict]
        来自 get_official_eval_result 的结构化评估数据。
    csv_path : str
        CSV 文件路径，若不存在则自动创建并写入表头。
    model_name : str
        模型/架构名称，如 'monodle', 'yolo3d_v8n' 等。
    epoch : int or None
        当前 epoch 编号。
    """
    import csv
    from datetime import datetime

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    # 固定列名（与 rich_data 中 metrics 的 key 对应）
    metric_keys = ["bbox", "bev", "3d", "aos"]
    difficulties = ["easy", "mod", "hard"]

    fieldnames = ["epoch", "timestamp", "model", "category", "overlap"]
    for mk in metric_keys:
        for diff in difficulties:
            fieldnames.append(f"{mk}_{diff}")
    for mk in metric_keys:
        for diff in difficulties:
            fieldnames.append(f"{mk}_R40_{diff}")

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for item in rich_data:
            row = {
                "epoch": epoch if epoch is not None else "",
                "timestamp": ts,
                "model": model_name,
                "category": item["class_name"],
                "overlap": item["overlap_str"],
            }
            # AP (11-point)
            for mk in metric_keys:
                scores = item["metrics"].get(mk)
                for i, diff in enumerate(difficulties):
                    col = f"{mk}_{diff}"
                    row[col] = f"{float(scores[i]):.4f}" if scores is not None and i < len(scores) else ""
            # AP R40
            for mk in metric_keys:
                scores = item["metrics_R40"].get(mk)
                for i, diff in enumerate(difficulties):
                    col = f"{mk}_R40_{diff}"
                    row[col] = f"{float(scores[i]):.4f}" if scores is not None and i < len(scores) else ""

            writer.writerow(row)

    logger.info(f"评估结果已追加保存到 {csv_path}")


def print_best_epoch_results(csv_path, metric_key='Car_3d_moderate_R40', logger_obj=None):
    """
    从 eval_results.csv 中找到最佳 epoch，并以论文表格格式打印该 epoch 的所有结果。

    输出格式与 KITTI benchmark 论文表格一致：
    - Table: Car 3D/BEV/AOS @ IoU=0.7 (test set style)
    - Table: Car 3D/BEV @ IoU=0.7 and IoU=0.5 (validation set style)

    Parameters
    ----------
    csv_path : str
        eval_results.csv 文件路径。
    metric_key : str
        用于确定最佳 epoch 的指标列名，如 '3d_R40_mod'。
    logger : MonoDDLELogger or None
        日志对象。
    """
    import csv

    if not os.path.exists(csv_path):
        if logger:
            logger.warning(f"CSV file not found: {csv_path}")
        return

    # 读取 CSV
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        if logger:
            logger.warning("CSV file is empty.")
        return

    # 将 metric_key 映射到 CSV 列名
    # metric_key 格式: Car_3d_moderate_R40 -> 需要在 category=Car, overlap 包含 0.70 的行中查找 3d_R40_mod
    # 解析 metric_key
    parts = metric_key.split('_')
    # e.g. Car_3d_moderate_R40 -> category=Car, metric=3d, difficulty=moderate, R40
    target_category = parts[0]
    is_r40 = metric_key.endswith('_R40')
    if is_r40:
        # e.g. Car_3d_moderate_R40
        difficulty_map = {'easy': 'easy', 'moderate': 'mod', 'hard': 'hard'}
        target_difficulty = difficulty_map.get(parts[-2], parts[-2])
        target_metric_type = '_'.join(parts[1:-2])  # e.g. '3d'
        csv_col = f"{target_metric_type}_R40_{target_difficulty}"
    else:
        difficulty_map = {'easy': 'easy', 'moderate': 'mod', 'hard': 'hard'}
        target_difficulty = difficulty_map.get(parts[-1], parts[-1])
        target_metric_type = '_'.join(parts[1:-1])
        csv_col = f"{target_metric_type}_{target_difficulty}"

    # 找到最佳 epoch
    best_epoch = None
    best_val = -1.0

    for row in rows:
        if row.get('category', '') != target_category:
            continue
        # 选择主要 overlap (0.70 for Car)
        overlap = row.get('overlap', '')
        if target_category == 'Car' and '0.70, 0.70, 0.70' not in overlap:
            continue
        elif target_category == 'Pedestrian' and '0.50, 0.50, 0.50' not in overlap:
            continue
        elif target_category == 'Cyclist' and '0.50, 0.50, 0.50' not in overlap:
            continue

        epoch_str = row.get('epoch', '')
        if not epoch_str:
            continue
        try:
            epoch_val = int(epoch_str)
        except (ValueError, TypeError):
            continue

        val_str = row.get(csv_col, '')
        if not val_str:
            continue
        try:
            val = float(val_str)
        except (ValueError, TypeError):
            continue

        if val > best_val:
            best_val = val
            best_epoch = epoch_val

    if best_epoch is None:
        logger.warning(f"Could not find best epoch for metric: {metric_key}")
        return

    # 收集最佳 epoch 的所有行
    best_rows = [r for r in rows if r.get('epoch', '') == str(best_epoch)]

    if not best_rows:
        logger.warning(f"No data found for best epoch {best_epoch}")
        return

    # 按 category 和 overlap 组织数据
    categories = {}
    for row in best_rows:
        cat = row.get('category', '')
        overlap = row.get('overlap', '')
        if cat not in categories:
            categories[cat] = {}
        categories[cat][overlap] = row

    model_name = best_rows[0].get('model', 'unknown')

    # ═══ 打印 Test Set 风格表格 (AP R40) ═══
    # 格式与图片中 Table 3/4 一致
    logger.success(f"Found Best Epoch: {best_epoch} (by {metric_key} = {best_val:.2f})")

    for cat_name in ['Car', 'Pedestrian', 'Cyclist']:
        if cat_name not in categories:
            continue

        overlaps = categories[cat_name]

        # 确定主要和次要 overlap
        if cat_name == 'Car':
            primary_overlap_key = '0.70, 0.70, 0.70'
            secondary_overlap_key = '0.50, 0.50, 0.50'
        elif cat_name == 'Pedestrian':
            primary_overlap_key = '0.50, 0.50, 0.50'
            secondary_overlap_key = '0.50, 0.25, 0.25'
        else:  # Cyclist
            primary_overlap_key = '0.50, 0.50, 0.50'
            secondary_overlap_key = '0.50, 0.25, 0.25'

        # === 统一结果表格 (与 README KITTI 验证集风格一致) ===
        primary_row = overlaps.get(primary_overlap_key)
        secondary_row = overlaps.get(secondary_overlap_key)

        if not primary_row:
            continue

        table = Table(
            title=f"{cat_name} AP_R40 Performance (KITTI Val Style)",
            header_style="bold cyan",
            border_style="magenta",
            box=box.MARKDOWN, # 使用 Markdown 风格边框，方便直接复制进入 README
        )
        
        table.add_column("Method", style="bold", no_wrap=True)
        
        # 定义列
        cols = [
            ("3D@0.7 (Easy)", "3d_R40_easy", primary_row),
            ("3D@0.7 (Mod.)", "3d_R40_mod", primary_row),
            ("3D@0.7 (Hard)", "3d_R40_hard", primary_row),
            ("BEV@0.7 (Easy)", "bev_R40_easy", primary_row),
            ("BEV@0.7 (Mod.)", "bev_R40_mod", primary_row),
            ("BEV@0.7 (Hard)", "bev_R40_hard", primary_row),
        ]
        
        if cat_name == 'Car':
            # 根据 README Table 2，增加 0.5 结果
            if secondary_row:
                cols.extend([
                    ("3D@0.5 (Easy)", "3d_R40_easy", secondary_row),
                    ("3D@0.5 (Mod.)", "3d_R40_mod", secondary_row),
                    ("3D@0.5 (Hard)", "3d_R40_hard", secondary_row),
                    ("BEV@0.5 (Easy)", "bev_R40_easy", secondary_row),
                    ("BEV@0.5 (Mod.)", "bev_R40_mod", secondary_row),
                    ("BEV@0.5 (Hard)", "bev_R40_hard", secondary_row),
                ])
            # 添加 AOS (Table 1 风格)
            cols.extend([
                ("AOS (Easy)", "aos_R40_easy", primary_row),
                ("AOS (Mod.)", "aos_R40_mod", primary_row),
                ("AOS (Hard)", "aos_R40_hard", primary_row),
            ])

        for label, _, _ in cols:
            style = "green" if "Easy" in label else "yellow" if "Mod." in label else "red"
            table.add_column(label, justify="right", style=style)

        def fmt(val_str):
            try:
                v = float(val_str)
                return f"{v:.2f}"
            except (ValueError, TypeError):
                return "-"

        row_data = [f"{model_name} (ep{best_epoch})"]
        for _, key, row in cols:
            row_data.append(fmt(row.get(key, '')))

        table.add_row(*row_data)
        console.print(table)
        _log_rich_content(table)

        console.print("")


# ============ 便捷导出 ============

__all__ = [
    'MonoDDLELogger',
    'console',
    'create_progress_bar',
    'create_epoch_progress',
    'print_kitti_eval_results',
    'save_eval_to_csv',
    'print_best_epoch_results',
]
