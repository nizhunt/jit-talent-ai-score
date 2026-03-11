from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ui" / "investor-mermaid-diagram.png"

W, H = 1600, 780
EXPORT_SCALE = 2
BG = (248, 244, 238)
INK = (29, 26, 23)
MUTED = (104, 95, 86)
ACCENT = (196, 81, 45)
GOLD = (217, 163, 41)
TEAL = (15, 118, 110)
CARD = (255, 251, 245)
CARD_WARM = (255, 244, 237)
CARD_TEAL = (239, 248, 246)
LINE = (226, 214, 198)
LINE_WARM = (228, 179, 159)
LINE_TEAL = (168, 216, 210)
SHADOW = (89, 60, 34, 18)


def load_font(paths: list[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for raw_path in paths:
        path = Path(raw_path)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(
    [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ],
    24,
)
FONT_BODY = load_font(
    [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ],
    17,
)
FONT_BODY_SMALL = load_font(
    [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ],
    16,
)
FONT_PILL = load_font(
    [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ],
    14,
)
FONT_METRIC = load_font(
    [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ],
    46,
)


img = Image.new("RGBA", (W, H), BG)
draw = ImageDraw.Draw(img)


def wrap(text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        words = para.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            test = word if not current else f"{current} {word}"
            if draw.textlength(test, font=font) <= width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def shadow_box(x: int, y: int, w: int, h: int, radius: int, fill: tuple[int, int, int], outline: tuple[int, int, int]) -> None:
    shadow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    shadow_draw.rounded_rectangle((x, y + 8, x + w, y + h + 8), radius=radius, fill=SHADOW)
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(10))
    img.alpha_composite(shadow_layer)
    draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline=outline, width=2)


def pill(x: int, y: int, label: str, color: tuple[int, int, int], fill: tuple[int, int, int], outline: tuple[int, int, int]) -> None:
    text_width = draw.textlength(label, font=FONT_PILL)
    draw.rounded_rectangle((x, y, x + text_width + 24, y + 28), radius=14, fill=fill, outline=outline, width=1)
    draw.text((x + 12, y + 6), label, font=FONT_PILL, fill=color)


def card_box(
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    body: str,
    label: str,
    label_color: tuple[int, int, int],
    *,
    fill: tuple[int, int, int] = CARD,
    outline: tuple[int, int, int] = LINE,
    body_font: ImageFont.ImageFont | None = None,
) -> None:
    body_font = body_font or FONT_BODY
    shadow_box(x, y, w, h, 24, fill, outline)
    pill(x + 14, y - 14, label, label_color, CARD, outline)
    yy = y + 22
    for line in wrap(title, FONT_TITLE, w - 34):
        draw.text((x + 18, yy), line, font=FONT_TITLE, fill=INK)
        yy += 30
    yy += 4
    step = 25 if body_font == FONT_BODY_SMALL else 27
    for line in wrap(body, body_font, w - 34):
        draw.text((x + 18, yy), line, font=body_font, fill=MUTED)
        yy += step


def line_arrow(points: list[tuple[int, int]], color: tuple[int, int, int], width: int = 6) -> None:
    for idx in range(len(points) - 1):
        draw.line((points[idx][0], points[idx][1], points[idx + 1][0], points[idx + 1][1]), fill=color, width=width)
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    head = 14
    p1 = (x2, y2)
    p2 = (
        x2 - head * math.cos(angle) + head * 0.55 * math.sin(angle),
        y2 - head * math.sin(angle) - head * 0.55 * math.cos(angle),
    )
    p3 = (
        x2 - head * math.cos(angle) - head * 0.55 * math.sin(angle),
        y2 - head * math.sin(angle) + head * 0.55 * math.cos(angle),
    )
    draw.polygon([p1, p2, p3], fill=color)


# Left column
card_box(
    40,
    96,
    300,
    218,
    "How the model scores fit",
    "Skills and tools fit: 35%\nRelevant experience: 35%\nResponsibilities overlap: 15%\nSeniority and leadership fit: 10%\nLogistics fit: 5%",
    "Scoring Weights",
    ACCENT,
    body_font=FONT_BODY_SMALL,
)

# Main flow
card_box(
    420,
    80,
    290,
    134,
    "Employer drops job on platform!",
    "First step.",
    "Input",
    ACCENT,
    fill=CARD_WARM,
    outline=LINE_WARM,
)
card_box(
    760,
    80,
    320,
    134,
    "Enters the JIT request queue",
    "The request is captured and routed into the Slack-driven workflow.",
    "Queue",
    ACCENT,
)
card_box(
    1130,
    80,
    350,
    134,
    "Candidate TAM",
    "The system expands the role and scans the market to size the candidate universe.",
    "Search",
    ACCENT,
)
card_box(
    620,
    294,
    470,
    190,
    "AI scoring agent",
    "Candidates are deduplicated and scored from 0 to 10 using weighted fit criteria, written reasoning, and outreach-ready personalization snippets.",
    "Scoring",
    TEAL,
    fill=CARD_TEAL,
    outline=LINE_TEAL,
)
card_box(
    1140,
    294,
    370,
    190,
    "Ranked sheet and recruiter review",
    "Results are delivered back as a Google Sheet in Slack, where the recruiter chooses the score threshold for outreach.",
    "Output",
    GOLD,
)
card_box(
    80,
    554,
    320,
    154,
    "Threshold reply",
    "Recruiter replies with a minimum score such as 5 to approve outreach.",
    "Decision",
    ACCENT,
    fill=CARD_WARM,
    outline=LINE_WARM,
)
card_box(
    450,
    554,
    340,
    154,
    "Email verification",
    "Direct emails are found, filtered, and checked for deliverability before activation.",
    "Enrichment",
    ACCENT,
)
card_box(
    840,
    538,
    320,
    186,
    "AI-personalized campaign per lead",
    "Each approved lead is enrolled with personalized outbound messaging.",
    "Activation",
    TEAL,
    fill=CARD_TEAL,
    outline=LINE_TEAL,
)

# Connectors
line_arrow([(710, 147), (760, 147)], ACCENT)
line_arrow([(1080, 147), (1130, 147)], ACCENT)
line_arrow([(1305, 214), (1305, 250), (855, 250), (855, 294)], TEAL)
line_arrow([(1090, 389), (1140, 389)], GOLD)
line_arrow([(1325, 484), (1325, 520), (240, 520), (240, 554)], ACCENT)
line_arrow([(400, 631), (450, 631)], ACCENT)
line_arrow([(790, 631), (840, 631)], TEAL)

OUT.parent.mkdir(parents=True, exist_ok=True)
final_image = img.convert("RGB")
final_image = final_image.resize((W * EXPORT_SCALE, H * EXPORT_SCALE), Image.Resampling.LANCZOS)
final_image.save(OUT, "PNG", optimize=True)
print(OUT)
