import os
import re
import csv
from pathlib import Path

def parse_log(log_path):
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Find the section for Car AP_R40@0.70, 0.70, 0.70
    match_70 = re.search(r'Car AP_R40@0\.70, 0\.70, 0\.70:\nbbox AP:(.*?)\nbev  AP:(.*?)\n3d   AP:(.*?)\naos  AP:(.*?)\n', content)
    
    # Find the section for Car AP_R40@0.70, 0.50, 0.50
    match_50 = re.search(r'Car AP_R40@0\.70, 0\.50, 0\.50:\nbbox AP:(.*?)\nbev  AP:(.*?)\n3d   AP:(.*?)\naos  AP:(.*?)\n', content)
    
    # Find Model Complexity
    complexity_match = re.search(r'Model Complexity \| Input: .*? \| FLOPs: (.*?) \| Params: (.*?)\n', content)
    flops = complexity_match.group(1) if complexity_match else 'N/A'
    params = complexity_match.group(2) if complexity_match else 'N/A'

    if match_70 and match_50:
        bbox_70 = [float(x.strip()) for x in match_70.group(1).split(',')]
        bev_70 = [float(x.strip()) for x in match_70.group(2).split(',')]
        ap3d_70 = [float(x.strip()) for x in match_70.group(3).split(',')]
        aos_70 = [float(x.strip()) for x in match_70.group(4).split(',')]
        
        bbox_50 = [float(x.strip()) for x in match_50.group(1).split(',')]
        bev_50 = [float(x.strip()) for x in match_50.group(2).split(',')]
        ap3d_50 = [float(x.strip()) for x in match_50.group(3).split(',')]
        aos_50 = [float(x.strip()) for x in match_50.group(4).split(',')]
        
        return {
            'bbox_easy_70': bbox_70[0], 'bbox_mod_70': bbox_70[1], 'bbox_hard_70': bbox_70[2],
            'bev_easy_70': bev_70[0], 'bev_mod_70': bev_70[1], 'bev_hard_70': bev_70[2],
            '3d_easy_70': ap3d_70[0], '3d_mod_70': ap3d_70[1], '3d_hard_70': ap3d_70[2],
            'aos_easy_70': aos_70[0], 'aos_mod_70': aos_70[1], 'aos_hard_70': aos_70[2],
            
            'bbox_easy_50': bbox_50[0], 'bbox_mod_50': bbox_50[1], 'bbox_hard_50': bbox_50[2],
            'bev_easy_50': bev_50[0], 'bev_mod_50': bev_50[1], 'bev_hard_50': bev_50[2],
            '3d_easy_50': ap3d_50[0], '3d_mod_50': ap3d_50[1], '3d_hard_50': ap3d_50[2],
            'aos_easy_50': aos_50[0], 'aos_mod_50': aos_50[1], 'aos_hard_50': aos_50[2],
            
            'flops': flops,
            'params': params,
        }
    return None

def main():
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / 'experiments' / 'results'
    results = []
    
    for log_file in results_dir.rglob('train.log'):
        # Path format: experiments/results/<exp_name>/<timestamp>/logs/train.log
        # We want to extract <exp_name>
        parts = log_file.relative_to(results_dir).parts
        if len(parts) >= 4 and parts[-2] == 'logs' and parts[-1] == 'train.log':
            exp_name = '/'.join(parts[:-3])
            timestamp = parts[-3]
            
            metrics = parse_log(log_file)
            if metrics:
                metrics['experiment'] = exp_name
                metrics['timestamp'] = timestamp
                results.append(metrics)
    
    # Sort by 3d_mod_70 descending
    results.sort(key=lambda x: x['3d_mod_70'], reverse=True)
    
    # Write CSV
    csv_path = project_root / 'summary.csv'
    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            'experiment', 
            '3d_easy_70', '3d_mod_70', '3d_hard_70', 
            'bev_easy_70', 'bev_mod_70', 'bev_hard_70', 
            '3d_easy_50', '3d_mod_50', '3d_hard_50', 
            'bev_easy_50', 'bev_mod_50', 'bev_hard_50', 
            'bbox_easy_70', 'bbox_mod_70', 'bbox_hard_70', 
            'aos_easy_70', 'aos_mod_70', 'aos_hard_70', 
            'bbox_easy_50', 'bbox_mod_50', 'bbox_hard_50', 
            'aos_easy_50', 'aos_mod_50', 'aos_hard_50', 
            'flops', 'params', 'timestamp'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    # Write Markdown
    md_path = project_root / 'summary.md'
    with open(md_path, 'w') as f:
        f.write('# Experiment Results\n\n')
        f.write('| Experiment | 3D AP@0.7 (E/M/H) | BEV AP@0.7 (E/M/H) | 3D AP@0.5 (E/M/H) | BEV AP@0.5 (E/M/H) | BBox AP (E/M/H) | AOS (E/M/H) | FLOPs | Params | Timestamp |\n')
        f.write('|---|---|---|---|---|---|---|---|---|---|\n')
        for r in results:
            f.write(f"| {r['experiment']} | {r['3d_easy_70']:.2f} / **{r['3d_mod_70']:.2f}** / {r['3d_hard_70']:.2f} | {r['bev_easy_70']:.2f} / {r['bev_mod_70']:.2f} / {r['bev_hard_70']:.2f} | {r['3d_easy_50']:.2f} / {r['3d_mod_50']:.2f} / {r['3d_hard_50']:.2f} | {r['bev_easy_50']:.2f} / {r['bev_mod_50']:.2f} / {r['bev_hard_50']:.2f} | {r['bbox_easy_70']:.2f} / {r['bbox_mod_70']:.2f} / {r['bbox_hard_70']:.2f} | {r['aos_easy_70']:.2f} / {r['aos_mod_70']:.2f} / {r['aos_hard_70']:.2f} | {r['flops']} | {r['params']} | {r['timestamp']} |\n")
            
    print(f"Saved {len(results)} results to {csv_path} and {md_path}")

if __name__ == '__main__':
    main()
