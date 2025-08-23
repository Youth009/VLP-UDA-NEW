import os
import cv2
import csv
import shutil

# 数据集根目录
root = "/home/og/wjj/wjj-research/Data/phyto_plankton/"
# 坏图保存根目录
bad_root = "bad_images"
# 报告文件
report_path = "bad_images_report.csv"

bad = []

for dirpath, _, files in os.walk(root):
    print("正在检查文件夹：", dirpath)
    for f in files:
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
            path = os.path.join(dirpath, f)
            try:
                img = cv2.imread(path)

                # 检查能否读取
                if img is None:
                    reason = "无法读取"
                    bad.append((path, reason))
                else:
                    # 检查通道数
                    if img.ndim != 3 or img.shape[2] != 3:
                        reason = f"通道数异常: {img.shape}"
                        bad.append((path, reason))
                    else:
                        # 检查分辨率
                        h, w, _ = img.shape
                        if h < 32 or w < 32:
                            reason = f"分辨率过小: {w}x{h}"
                            bad.append((path, reason))
                        else:
                            continue

                # 生成坏图保存路径（保持原始目录结构）
                rel_path = os.path.relpath(path, root)  # 相对路径
                new_path = os.path.join(bad_root, rel_path)
                os.makedirs(os.path.dirname(new_path), exist_ok=True)

                shutil.move(path, new_path)

            except Exception as e:
                reason = f"读取异常: {str(e)}"
                bad.append((path, reason))

                rel_path = os.path.relpath(path, root)
                new_path = os.path.join(bad_root, rel_path)
                os.makedirs(os.path.dirname(new_path), exist_ok=True)

                shutil.move(path, new_path)

# 打印结果
print("坏图数量:", len(bad))
for b in bad[:20]:
    print("坏图:", b)

# 保存报告 CSV
if bad:
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["原始路径", "问题类型"])
        writer.writerows(bad)
    print(f"详细报告已保存到 {report_path}")
    print(f"坏图已移动到 {bad_root}/ 对应子文件夹中")
else:
    print("未发现坏图 ✅")
