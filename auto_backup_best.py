#!/usr/bin/env python3
"""
auto_backup_best.py — 训练期间自动监控 eval_loss, 一旦发现新的历史最低点,
立刻把对应 checkpoint 的模型权重(跳过优化器状态大文件)复制到独立备份目录,
不依赖 save_total_limit 的滚动机制, 不需要人工盯着手动干预。

用法:
  python auto_backup_best.py --log_path <trainer_log.jsonl路径> \
      --ckpt_dir <checkpoint所在训练输出目录> \
      --backup_dir <独立备份目标目录> \
      --check_interval 300
"""
import argparse, json, os, shutil, time

def get_best_step(log_path):
    """扫描 trainer_log.jsonl, 返回 (best_step, best_eval_loss) 或 (None, None)"""
    best_step, best_loss = None, float('inf')
    if not os.path.exists(log_path):
        return None, None
    with open(log_path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get('eval_loss') is not None:
                if d['eval_loss'] < best_loss:
                    best_loss = d['eval_loss']
                    best_step = d.get('current_steps')
    return best_step, best_loss

def backup_checkpoint(src, dst):
    """只复制模型权重和配置文件, 跳过优化器状态(global_step*目录)等大文件"""
    os.makedirs(dst, exist_ok=True)
    copied = 0
    for fname in os.listdir(src):
        fpath = os.path.join(src, fname)
        if os.path.isdir(fpath):
            continue  # 跳过 global_stepN 这类优化器状态子目录
        if fname.endswith(('.safetensors', '.json', '.txt', '.jinja')):
            shutil.copy2(fpath, os.path.join(dst, fname))
            copied += 1
    return copied

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log_path', required=True)
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--backup_dir', required=True)
    ap.add_argument('--check_interval', type=int, default=300)
    args = ap.parse_args()

    os.makedirs(args.backup_dir, exist_ok=True)
    backed_up_step = None

    print(f"[auto_backup] 开始监控 {args.log_path}, 每 {args.check_interval}s 检查一次", flush=True)

    while True:
        try:
            best_step, best_loss = get_best_step(args.log_path)
            if best_step is not None and best_step != backed_up_step:
                src = os.path.join(args.ckpt_dir, f"checkpoint-{best_step}")
                if os.path.isdir(src):
                    dst = os.path.join(args.backup_dir, f"checkpoint-{best_step}-eval{best_loss:.6f}")
                    if not os.path.exists(dst):
                        print(f"[auto_backup] 发现新最佳: step={best_step}, eval_loss={best_loss:.6f}, 开始备份...", flush=True)
                        try:
                            n = backup_checkpoint(src, dst)
                            print(f"[auto_backup] 备份完成, 共{n}个文件 -> {dst}", flush=True)
                            backed_up_step = best_step
                            # 清理之前备份的旧版本(不是当前最新最佳的), 避免累积占用空间
                            for old in os.listdir(args.backup_dir):
                                old_path = os.path.join(args.backup_dir, old)
                                if old_path != dst and old.startswith('checkpoint-'):
                                    print(f"[auto_backup] 清理旧备份: {old_path}", flush=True)
                                    shutil.rmtree(old_path, ignore_errors=True)
                        except Exception as e:
                            print(f"[auto_backup] 备份失败: {e}", flush=True)
                else:
                    print(f"[auto_backup] 最佳checkpoint({src})尚未生成或已被清理, 等待下一轮", flush=True)
        except Exception as e:
            print(f"[auto_backup] 主循环异常(不退出, 继续监控): {e}", flush=True)
        time.sleep(args.check_interval)

if __name__ == '__main__':
    main()