"""Local UI regression using already observed accessible controls."""
from pathlib import Path
import subprocess,json,hashlib
root=Path(__file__).resolve().parents[1]
cli=r'D:\npm_global\npm-cache\_npx\31e32ef8478fbf80\node_modules\@playwright\cli\playwright-cli.js'
code='''async page => {
 await page.getByRole("button",{name:"刷新任务列表"}).click();
 const labels = await page.getByRole("button").allTextContents();
 const label = labels.find(s=>s.trim().startsWith("纳西妲 ·") && s.includes("处理完成"));
 if (!label) throw new Error("No completed character task in visible task list");
 await page.getByRole("button",{name:/^纳西妲 ·.*处理完成/}).first().click();
 const preview=page.getByRole("img",{name:"目标角色实际 Blender 静帧预览"});
 await preview.scrollIntoViewIfNeeded();
 await preview.evaluate(img=>img.decode());
 const image=await preview.evaluate(img=>({width:img.naturalWidth,height:img.naturalHeight}));
 await page.screenshot({path:"output/playwright/v050-character-desktop.png",fullPage:true});
 const pending=page.waitForEvent("download");
 await page.getByRole("link",{name:"下载纳西妲 Blender 工程 ↗"}).click();
 const download=await pending;
 await download.saveAs("D:/codex/danap/MotionBridge/output/playwright/v050-nahida.blend");
 await page.setViewportSize({width:390,height:844});
 const mobile=await page.evaluate(()=>({viewport:innerWidth,content:document.documentElement.scrollWidth}));
 await page.screenshot({path:"output/playwright/v050-character-mobile.png",fullPage:true});
 await page.setViewportSize({width:1440,height:1000});
 await page.getByRole("button",{name:"返回源视频任务 ↗"}).click();
 await page.waitForFunction(()=>[...document.querySelectorAll("button")].some(b=>b.textContent.trim()==="生成角色工程"&&!b.disabled),null,{timeout:10000});
 const toggle=page.getByRole("checkbox",{name:"手脚增强 按画面对齐手臂、连续跟踪手腕与支撑脚"});
 await toggle.uncheck(); const canDisable=!(await toggle.isChecked()); await toggle.check();
 return {selected:label,image,mobile,canDisable,enhancementChecked:await toggle.isChecked(),downloadFailure:await download.failure(),sourceReturned:true};
}'''
r=subprocess.run(['node',cli,'-s=motionbridge-v050','run-code',code],cwd=root,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
(root/'data/release-v050/browser-validation.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
print(r.stdout,r.stderr)
assert r.returncode==0 and '### Error' not in r.stdout
ids=json.loads((root/'data/release-v050/final_jobs.json').read_text())
def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
expected=digest(root/'data/jobs'/ids['character']/'outputs/Nahida_MotionBridge_Dance.blend')
actual=digest(root/'output/playwright/v050-nahida.blend')
assert actual==expected
(root/'data/release-v050/browser-download.json').write_text(json.dumps({'matches':True,'sha256':actual,'job':ids['character']}),encoding='utf-8')
subprocess.run(['node',cli,'-s=motionbridge-v050','close'],cwd=root,check=True,timeout=30)
