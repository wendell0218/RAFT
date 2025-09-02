from PIL import Image, ImageDraw, ImageFont, ImageSequence
import os

def concat_gifs(gif1_path, gif2_path, output_path="output.gif", font_path=None, font_size=20):
    gif1 = Image.open(gif1_path)
    gif2 = Image.open(gif2_path)

    name1 = os.path.basename(gif1_path)
    name2 = os.path.basename(gif2_path)

    w1, h1 = gif1.size
    w2, h2 = gif2.size

    if font_path is not None:
        font = ImageFont.truetype(font_path, font_size)
    else:
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

    dummy_img = Image.new("RGB", (10, 10))
    dummy_draw = ImageDraw.Draw(dummy_img)
    text_h = max(
        dummy_draw.textbbox((0, 0), name1, font=font)[3],
        dummy_draw.textbbox((0, 0), name2, font=font)[3]
    )
    height = max(h1, h2) + text_h + 10
    width = w1 + w2

    frames = []
    for frame1, frame2 in zip(ImageSequence.Iterator(gif1), ImageSequence.Iterator(gif2)):
        f1 = frame1.convert("RGBA")
        f2 = frame2.convert("RGBA")

        new_frame = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        draw = ImageDraw.Draw(new_frame)

        bbox1 = draw.textbbox((0, 0), name1, font=font)
        bbox2 = draw.textbbox((0, 0), name2, font=font)
        text_w1 = bbox1[2] - bbox1[0]
        text_w2 = bbox2[2] - bbox2[0]

        draw.text((w1 // 2 - text_w1 // 2, 0), name1, fill="black", font=font)
        draw.text((w1 + w2 // 2 - text_w2 // 2, 0), name2, fill="black", font=font)

        new_frame.paste(f1, (0, text_h + 5), f1)
        new_frame.paste(f2, (w1, text_h + 5), f2)

        frames.append(new_frame)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=gif1.info.get("duration", 100),
        loop=0,
        disposal=2
    )

concat_gifs(
    "normal.gif",
    "remove_bg.gif",
    "merged.gif",
    font_size=32 
)
