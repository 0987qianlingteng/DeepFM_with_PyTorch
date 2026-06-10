from pathlib import Path
import csv
import textwrap

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

FONT = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")
FONT_MONO = Path(r"C:\Windows\Fonts\consola.ttf")

ZH = FontProperties(fname=str(FONT), size=12)
ZH_BOLD = FontProperties(fname=str(FONT_BOLD), size=12)
MONO = FontProperties(fname=str(FONT_MONO), size=10)

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "Microsoft YaHei", "SimHei"]

ACCENT = "#0f766e"
ACCENT2 = "#b45309"
INK = "#1f2933"
MUTED = "#64748b"
PAPER = "#fffaf0"
LINE = "#cbd5e1"
GREEN = "#e6f4ef"
WARM = "#fff4de"
BLUE = "#eaf3ff"


def text(ax, x, y, s, size=12, color=INK, ha="center", va="center", bold=False, mono=False, **kwargs):
    fp = MONO if mono else (ZH_BOLD if bold else ZH)
    ax.text(x, y, s, fontproperties=fp, fontsize=size, color=color, ha=ha, va=va, **kwargs)


def wrap_label(label, width=12):
    lines = []
    for line in str(label).splitlines():
        if len(line) <= width:
            lines.append(line)
        else:
            lines.extend(textwrap.wrap(line, width=width, break_long_words=False, replace_whitespace=False))
    return "\n".join(lines)


def box(ax, xy, wh, label, fc=PAPER, ec=LINE, size=12, wrap=12):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.03",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    text(ax, x + w / 2, y + h / 2, wrap_label(label, wrap), size=size, bold=True, linespacing=1.35)
    return patch


def arrow(ax, start, end, color=MUTED):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.5,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def finish(fig, name):
    out = FIG / name
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def draw_deepfm():
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    text(ax, 0.5, 0.96, "DeepFM 模型原理结构图", size=20, bold=True)
    box(ax, (0.04, 0.48), (0.15, 0.16), "输入特征 Xi / Xv\n用户字段 + 电影字段", BLUE, size=11, wrap=10)
    box(ax, (0.25, 0.48), (0.17, 0.16), "共享 Embedding\n低维稠密向量表示", GREEN, size=11, wrap=11)
    box(ax, (0.49, 0.69), (0.17, 0.14), "FM 分支\n一阶项 + 二阶交互", WARM)
    box(ax, (0.49, 0.35), (0.17, 0.14), "DNN 分支\n高阶非线性组合", WARM)
    box(ax, (0.75, 0.52), (0.12, 0.13), "融合 Logit\n相加", GREEN)
    box(ax, (0.75, 0.32), (0.12, 0.11), "Sigmoid\n兴趣概率", BLUE)
    box(ax, (0.75, 0.15), (0.12, 0.10), "Top-N 排序\n电影推荐", PAPER, size=11, wrap=10)
    for start, end in [
        ((0.19, 0.56), (0.25, 0.56)),
        ((0.42, 0.60), (0.49, 0.76)),
        ((0.42, 0.52), (0.49, 0.42)),
        ((0.66, 0.76), (0.75, 0.60)),
        ((0.66, 0.42), (0.75, 0.56)),
        ((0.81, 0.52), (0.81, 0.43)),
        ((0.81, 0.32), (0.81, 0.25)),
    ]:
        arrow(ax, start, end)
    note = FancyBboxPatch(
        (0.05, 0.12),
        0.44,
        0.18,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=LINE,
        facecolor="#f8fafc",
    )
    ax.add_patch(note)
    text(ax, 0.075, 0.255, "核心思想", size=13, color=ACCENT, ha="left", bold=True)
    text(
        ax,
        0.075,
        0.195,
        "FM 负责低阶交互记忆，DNN 负责高阶组合泛化；\n两条分支共享 Embedding，并在输出层联合训练。",
        size=11.5,
        color=MUTED,
        ha="left",
        linespacing=1.6,
    )
    return finish(fig, "report_deepfm_architecture.png")


def draw_pipeline():
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    text(ax, 0.5, 0.95, "MovieLens 1M 数据处理流程图", size=20, bold=True)
    steps = [
        ("原始数据读取\nratings / users / movies", 0.05, 0.67, BLUE),
        ("表连接与字段提取\n用户属性 + 电影类型 + 年份", 0.30, 0.67, GREEN),
        ("类别特征编码\nuser_idx / movie_idx / genre_idx", 0.55, 0.67, WARM),
        ("隐式反馈建模\nrating >= 4 为正样本", 0.05, 0.32, BLUE),
        ("负采样\nneg_ratio = 1", 0.30, 0.32, GREEN),
        ("训练/验证划分\n保存 npz + meta + csv", 0.55, 0.32, WARM),
        ("DeepFM 训练与验证\nAUC / Loss / Time", 0.79, 0.49, PAPER),
    ]
    for label, x, y, fc in steps:
        box(ax, (x, y), (0.17, 0.15), label, fc, size=10.5, wrap=11)
    for start, end in [
        ((0.22, 0.745), (0.30, 0.745)),
        ((0.47, 0.745), (0.55, 0.745)),
        ((0.135, 0.67), (0.135, 0.47)),
        ((0.22, 0.395), (0.30, 0.395)),
        ((0.47, 0.395), (0.55, 0.395)),
        ((0.72, 0.395), (0.79, 0.54)),
        ((0.72, 0.745), (0.79, 0.58)),
    ]:
        arrow(ax, start, end)
    text(
        ax,
        0.5,
        0.13,
        "输出文件用于训练、评估和推荐推理：训练集/验证集、电影元数据、用户画像、历史正反馈集合。",
        size=12,
        color=MUTED,
    )
    return finish(fig, "report_data_pipeline.png")


def draw_topn():
    fig, ax = plt.subplots(figsize=(13.5, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    text(ax, 0.5, 0.95, "Top-N 个性化推荐推理流程图", size=20, bold=True)
    items = [
        ("输入测试 ID", 0.05, 0.60, BLUE),
        ("读取用户画像\n性别 / 年龄 / 职业 / 邮编", 0.27, 0.60, GREEN),
        ("候选电影集合\n过滤已正反馈电影", 0.49, 0.60, WARM),
        ("构造 DeepFM 输入\n用户特征 + 电影特征", 0.71, 0.60, GREEN),
        ("批量预测得分\nSigmoid 概率", 0.27, 0.27, BLUE),
        ("降序排序", 0.49, 0.27, WARM),
        ("输出 Top-N\n电影名 / 类型 / 分数", 0.71, 0.27, PAPER),
    ]
    for label, x, y, fc in items:
        box(ax, (x, y), (0.18, 0.15), label, fc, size=10.2, wrap=9)
    for start, end in [
        ((0.23, 0.675), (0.27, 0.675)),
        ((0.45, 0.675), (0.49, 0.675)),
        ((0.67, 0.675), (0.71, 0.675)),
        ((0.80, 0.60), (0.36, 0.42)),
        ((0.45, 0.345), (0.49, 0.345)),
        ((0.67, 0.345), (0.71, 0.345)),
    ]:
        arrow(ax, start, end)
    note = FancyBboxPatch(
        (0.14, 0.08),
        0.72,
        0.10,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=LINE,
        facecolor="#f8fafc",
    )
    ax.add_patch(note)
    text(ax, 0.5, 0.13, "推理阶段不重新训练模型，直接加载最佳权重，对候选电影批量打分并排序。", size=11.2, color=MUTED)
    return finish(fig, "report_topn_flow.png")


def draw_curve():
    rows = []
    with (ROOT / "ml1m_dual_target_try5.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]
    train = [float(r["train_loss"]) for r in rows]
    val = [float(r["val_loss"]) for r in rows]
    acc = [float(r["val_acc"]) for r in rows]
    auc = [float(r["val_auc"]) for r in rows]
    times = [float(r["epoch_time_sec"]) for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle("DeepFM 训练过程曲线", fontproperties=ZH_BOLD, fontsize=20, color=INK)
    axes[0].plot(epochs, train, marker="o", color=ACCENT, label="train_loss")
    axes[0].plot(epochs, val, marker="s", color=ACCENT2, label="val_loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, auc, marker="o", color=ACCENT, label="val_auc")
    axes[1].plot(epochs, acc, marker="s", color=ACCENT2, label="val_acc")
    axes[1].set_ylabel("Metric")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    best = max(range(len(auc)), key=lambda i: auc[i])
    axes[1].annotate(
        f"best AUC={auc[best]:.6f}\nepoch {epochs[best]}",
        xy=(epochs[best], auc[best]),
        xytext=(epochs[best] - 5, auc[best] - 0.08),
        arrowprops=dict(arrowstyle="->", color=INK),
        fontsize=11,
    )

    axes[2].bar(epochs, times, color="#8ec7bd")
    axes[2].axhline(10, color="#b91c1c", linestyle="--", linewidth=1.2, label="10s target")
    axes[2].set_ylabel("Time(s)")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return finish(fig, "report_epoch_training_curve.png")


def draw_cli():
    fig, ax = plt.subplots(figsize=(14, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.03, 0.06), 0.94, 0.88, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor="#111827", edgecolor="#111827"))
    ax.add_patch(FancyBboxPatch((0.03, 0.88), 0.94, 0.06, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor="#0b1220", edgecolor="#0b1220"))
    text(ax, 0.055, 0.905, "PowerShell - DeepFM Top-N 推荐演示", color="#e5e7eb", size=12, ha="left")
    lines = [
        "PS > python main.py ml_recommend --model_path ml1m_dual_target_try5.pt --task_type implicit --neg_ratio 1",
        "     --max_rows 500000 --embedding_size 24 --hidden_dims 256,128 --test_id 25 --top_n 5",
        "",
        "输入测试ID: 25",
        "对应用户: 1182 | 当前样本电影: Fantasia (1940) (Animation|Children's|Musical)",
        "Top-5 电影推荐:",
        "rank    title                                             genres                              score",
        "1       Nightmare Before Christmas, The (1993)              Children's|Comedy|Musical           1.000000",
        "2       Babe (1995)                                         Children's|Comedy|Drama             1.000000",
        "3       Men in Black (1997)                                 Action|Adventure|Comedy|Sci-Fi      1.000000",
        "4       Wizard of Oz, The (1939)                            Adventure|Children's|Drama|Musical  1.000000",
        "5       South Park: Bigger, Longer and Uncut (1999)         Animation|Comedy                    1.000000",
    ]
    y = 0.83
    for line in lines:
        fp = ZH if any("\u4e00" <= ch <= "\u9fff" for ch in line) else MONO
        ax.text(0.06, y, line, fontproperties=fp, fontsize=9.7, color=("#facc15" if line.startswith("PS >") or line.startswith("     --") else "#d1d5db"), ha="left", va="top")
        y -= 0.058
    return finish(fig, "report_cli_recommend_screenshot.png")


def draw_ui():
    fig, ax = plt.subplots(figsize=(14, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="square,pad=0", facecolor="#f7f0df", edgecolor="none"))
    text(ax, 0.07, 0.90, "DeepFM / MovieLens 1M", color=ACCENT2, size=12, ha="left", bold=True)
    text(ax, 0.07, 0.83, "电影个性化推荐演示", size=28, ha="left", bold=True)
    text(ax, 0.07, 0.77, "输入测试样本 ID，系统实时生成 Top-N 推荐电影列表。", color=MUTED, size=13, ha="left")

    metric_x = [0.63, 0.75, 0.87]
    for x, (value, label) in zip(metric_x, [("0.8132", "Best AUC"), ("3706", "Movies"), ("50000", "Test Samples")]):
        box(ax, (x, 0.78), (0.09, 0.11), f"{value}\n{label}", "white", size=11, wrap=12)

    control = FancyBboxPatch((0.06, 0.63), 0.88, 0.095, boxstyle="round,pad=0.025,rounding_size=0.025", linewidth=1.3, edgecolor=LINE, facecolor="white")
    ax.add_patch(control)
    text(ax, 0.19, 0.678, "测试 ID: 25", size=13, ha="center", bold=True)
    text(ax, 0.38, 0.678, "推荐数量: Top-5", size=13, ha="center", bold=True)
    text(ax, 0.58, 0.678, "生成推荐", size=13, ha="center", bold=True, color=ACCENT)
    text(ax, 0.75, 0.678, "随机样本", size=13, ha="center", bold=True, color=ACCENT2)

    sample = FancyBboxPatch((0.06, 0.20), 0.28, 0.34, boxstyle="round,pad=0.025,rounding_size=0.025", linewidth=1.3, edgecolor=LINE, facecolor="white")
    ax.add_patch(sample)
    text(ax, 0.20, 0.485, "当前测试样本", size=13, bold=True)
    text(ax, 0.20, 0.395, "Fantasia\n(1940)", size=15, bold=True)
    text(ax, 0.20, 0.305, "Animation | Children's | Musical", size=11.5, bold=True)
    text(ax, 0.20, 0.250, "用户 1182", size=13, bold=True)

    panel = FancyBboxPatch((0.38, 0.16), 0.56, 0.42, boxstyle="round,pad=0.025,rounding_size=0.025", linewidth=1.3, edgecolor=LINE, facecolor="white")
    ax.add_patch(panel)
    text(ax, 0.415, 0.535, "Top-N 推荐结果", size=14, ha="left", bold=True)
    movies = [
        ("1", "Nightmare Before Christmas, The (1993)", "Children's|Comedy|Musical", "1.000000"),
        ("2", "Babe (1995)", "Children's|Comedy|Drama", "1.000000"),
        ("3", "Men in Black (1997)", "Action|Adventure|Comedy|Sci-Fi", "1.000000"),
        ("4", "Wizard of Oz, The (1939)", "Adventure|Children's|Drama|Musical", "1.000000"),
        ("5", "South Park: Bigger, Longer and Uncut (1999)", "Animation|Comedy", "1.000000"),
    ]
    y = 0.475
    for rank, title, genre, score in movies:
        ax.add_patch(FancyBboxPatch((0.415, y - 0.042), 0.48, 0.055, boxstyle="round,pad=0.01,rounding_size=0.01", facecolor="#fffaf0", edgecolor=LINE))
        text(ax, 0.445, y - 0.014, rank, size=10.5, color=ACCENT, bold=True)
        text(ax, 0.472, y + 0.000, title, size=9.8, ha="left", bold=True)
        text(ax, 0.472, y - 0.026, genre.replace("|", " | "), size=8.7, color=MUTED, ha="left")
        text(ax, 0.870, y - 0.014, score, size=9.8, color=ACCENT2, ha="right", bold=True)
        y -= 0.067
    return finish(fig, "report_ui_demo_screenshot.png")


def write_captions():
    content = """报告插图清单与图题说明

图 3-1 DeepFM 模型原理结构图：展示输入特征、共享 Embedding、FM 分支、DNN 分支、融合层和 Top-N 排序之间的关系。
文件：figures/report_deepfm_architecture.png

图 4-1 MovieLens 1M 数据处理流程图：展示原始数据读取、字段合并、类别编码、隐式反馈建模、负采样和训练集/验证集保存流程。
文件：figures/report_data_pipeline.png

图 4-2 Top-N 个性化推荐推理流程图：展示输入测试 ID 后，从用户画像读取、候选过滤、模型打分到输出推荐列表的完整流程。
文件：figures/report_topn_flow.png

图 5-1 DeepFM 训练过程曲线：展示 train_loss、val_loss、val_auc、val_acc 和 epoch_time 随训练轮次变化的情况，并标注最佳 AUC 轮次。
文件：figures/report_epoch_training_curve.png

图 5-2 命令行 Top-5 推荐结果截图：展示通过 ml_recommend 指令输入 test_id=25 后输出真实电影标题、类型和预测分数的结果。
文件：figures/report_cli_recommend_screenshot.png

图 5-3 交互式网页推荐演示截图：展示本地网页中输入测试 ID、查看模型指标和 Top-N 推荐卡片的效果。
文件：figures/report_ui_demo_screenshot.png

建议插入位置：
1. DeepFM 模型原理结构图放在第 3 章“DeepFM 总体架构”之后。
2. 数据处理流程图放在第 4 章“数据预处理流程”之后。
3. Top-N 推荐推理流程图放在第 4 章“Top-N 推荐推理算法”之后。
4. 训练过程曲线放在第 5 章“训练过程分析”之后。
5. 命令行推荐结果截图和网页推荐演示截图放在第 5 章“Top-N 推荐演示结果”之后。
"""
    (ROOT / "report_figure_captions.txt").write_text(content, encoding="utf-8")


if __name__ == "__main__":
    outputs = [draw_deepfm(), draw_pipeline(), draw_topn(), draw_curve(), draw_cli(), draw_ui()]
    write_captions()
    for item in outputs:
        print(item.name)
    print("report_figure_captions.txt")
