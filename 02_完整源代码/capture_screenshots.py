"""使用本机 Edge 真正渲染推理效果页并截图，不绘制伪终端或伪指标。"""
import json
from pathlib import Path
from layout import BUNDLE, CHECKPOINTS, RESULTS, HISTORIES, PREDICTIONS, FIGURES, VECTOR, DEMO, DATA
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent


def main():
    entries=json.loads((DEMO / 'pages.json').read_text(encoding='utf-8'))
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        page=browser.new_page(viewport={'width':1540,'height':1100},device_scale_factor=3)
        for entry in entries:
            # 使用当前项目的页面文件，项目移动到桌面后仍可复现。
            page.goto((DEMO/f'{entry["stem"]}.html').as_uri())
            page.evaluate('document.fonts.ready')
            target=FIGURES/f'{entry["stem"]}.png'
            page.locator('main').screenshot(path=str(target))
            # 仅写入300DPI物理分辨率元数据，图像像素保持原始截图结果。
            with Image.open(target) as image:
                image.save(target,dpi=(300,300))
            print('真实页面截图：'+entry['stem'],flush=True)
        browser.close()
    manifest=[]
    for path in sorted((FIGURES).glob('*.png')):
        with Image.open(path) as im:
            manifest.append({'file':path.name,'width_px':im.width,'height_px':im.height,'dpi':list(im.info.get('dpi',[])),
                             'source':'trained-checkpoint inference page screenshot' if path.name.startswith(('fig13','fig14','fig15','fig16')) else 'recorded experiment data or method diagram'})
    (FIGURES / 'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
