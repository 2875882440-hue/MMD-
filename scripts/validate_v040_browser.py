"""Run observed local UI actions through the existing Playwright CLI session."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
cli=r'D:\npm_global\npm-cache\_npx\31e32ef8478fbf80\node_modules\@playwright\cli\playwright-cli.js'
code='''async page => {
 await page.getByRole("button", {name:"纳西妲 · user-dance-30s.mp4 · 新版重算 · 新版重算 · 新版重算 处理完成 · 10/23 20:01",exact:true}).click();
 const preview = page.getByRole("img", {name:"目标角色实际 Blender 静帧预览"});
 await preview.scrollIntoViewIfNeeded();
 const image = await preview.evaluate(img => ({complete:img.complete,width:img.naturalWidth,height:img.naturalHeight}));
 await page.screenshot({path:"output/playwright/v040-character-desktop.png",fullPage:true});
 const pending = page.waitForEvent("download");
 await page.getByRole("link",{name:"下载纳西妲 Blender 工程 ↗"}).click();
 const download = await pending;
 await download.saveAs("D:/codex/danap/MotionBridge/output/playwright/v040-nahida.blend");
 await page.setViewportSize({width:390,height:844});
 const mobile = await page.evaluate(() => ({viewport:innerWidth,client:document.documentElement.clientWidth,content:document.documentElement.scrollWidth}));
 await page.screenshot({path:"output/playwright/v040-character-mobile.png",fullPage:true});
 await page.setViewportSize({width:1440,height:1000});
 await page.getByRole("button",{name:"返回源视频任务 ↗"}).click();
 await page.waitForFunction(() => [...document.querySelectorAll("button")].some(b => b.textContent.trim()==="生成角色工程" && !b.disabled), null, {timeout:10000});
 const source = await page.getByRole("button",{name:"生成角色工程",exact:true}).isEnabled();
 return {image,mobile,sourceReturned:source,downloadFailure:await download.failure(),enhancementChecked:await page.getByRole("checkbox",{name:"手脚增强 辅助手腕、估算支撑脚；尽量保留可信快动作"}).isChecked()};
}'''
r=subprocess.run(['node',cli,'-s=motionbridge-v030-local','run-code',code],cwd=root,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
(root/'data/release-v040/browser-validation.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
print(r.stdout,r.stderr)
if r.returncode or '### Error' in r.stdout:
    raise RuntimeError('Browser check failed')
subprocess.run(['node',cli,'-s=motionbridge-v030-local','close'],cwd=root,check=True,timeout=30)
