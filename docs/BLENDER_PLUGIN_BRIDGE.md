# 从工作台使用 Blender 的 MMD Tools

## 怎么用

1. 在工作台的「替换到已有角色」中生成远坂凛动作工程，或选用已经生成的凛工程。
2. 在「衣发动力学」打开「启用衣发动力学」，选角色和动作工程，然后在「处理范围」选择「Blender 插件 · MMD Tools 衣服与稳发」。
3. 点击「开始衣发处理」。工作台自动启动隔离版 Blender，读取角色原始 PMX 中作者设置的物理。发束沿用此前确认的稳发处理。每个任务有自己的结果目录。
4. 查看「施力前 / 施力后」关键帧。要带走完整结果，勾选格式验证确认，选「Blender 工程」，导出并下载 ZIP。整包解压，保持目录结构后打开 `blender/scene.blend`。旁边的 MDD 记录衣发每个顶点的运动，单独拷贝工程会丢失这部分效果。

## 哪些工程适用

目前支持工作台已配置的远坂凛模型，使用单一活动骨骼动作、匹配原 PMX 的网格和顶点顺序。工作台已经处理过的顶点缓存不能再次叠加。新生成的合格凛工程会出现在动作列表里。其他角色需要独立匹配其原模型和骨架。本轮验证 Blender 工程包。骨骼 FBX 只有骨骼动画，无法携带这份顶点缓存。

## 插件从哪里读

本机服务读取项目 `config.json` 的 `blender_plugins.mmd_tools.blender_path`。指向已启用 MMD Tools 的便携 Blender 可执行文件。当前使用 Blender 5.2.4 LTS 和 MMD Tools 4.5.14。服务检查同目录的 `portable/extensions/user_default/mmd_tools/blender_manifest.toml`，运行时另查 Blender 主次版本与扩展是否启用。插件选择仅对当前任务有效，初始衣发开关关闭。主 Blender 继续用于工作台原有预览与打包。

## 软件接口与失败处理

- `GET /api/secondary/catalog` 提供插件文件是否就绪、原因、回退策略、角色和工作台已有动作工程的 ID。就绪只表示发现本地安装，运行时校验和回读另行完成。
- `POST /api/secondary/runs` 提交 `{"enabled":true,"profile":"rin","mode":"mmd_tools","source_id":"工作台返回的 ID","preview":true}`，返回任务 ID。服务端只接受本地登记的工程，客户端不能传插件路径或任意脚本。
- `GET /api/secondary/runs/{id}` 返回阶段、状态、实际处理方和预览。插件先在独立进程烘焙，以半帧间隔把结果写为 MDD。烘焙进程写完并关闭文件后直接退出，避开 Blender 5.2.4 关闭 Bullet 刚体场景时可能出现的原生崩溃。另一个干净进程验证输入和烘焙文件指纹，将缓存绑定到源工程副本，保存再打开，核对 48 根主骨、受保护身体顶点、动作曲线、蒙皮与形态键。动态顶点的回读误差门限为 0.000002 米。
- 缺插件或处理失败时，默认记录 `fallback_reason`，转回内置头发处理，衣物沿用输入。可在本机配置中设置 `fallback:false` 使任务明确报错，或把 `fallback_mode` 设为 `surface` 转入已有实验衣物流程，继续遵守原流程的拒绝条件。取消任务或检测到输入变化时停止。未经保护校验的插件文件不会作为结果提供下载。
- 处理成功后调用 `POST /api/secondary/runs/{id}/export`，请求体为 `{"formats":["blender"],"experimental":true}`。导出任务通过校验后，可用其 ID 访问 `GET /api/secondary/runs/{导出任务ID}/download`。

保存回读验证的是角色主动作和导出内容。衣发在快速动作或特定视角下是否穿插，仍需结合画面判断。

## 本机验证

实际工作台使用许佳琪 60–63 秒的远坂凛工程。插件任务 `4d3107eafaef442e87ed75a6a665c536` 完成 90 帧、179 个整数帧与半帧采样。48 根主骨逐采样相等；保存回读后的动态顶点最大坐标差 0 米。前后同相机预览 5 对。实际导出任务 `9a9e64cf74114f85b30e1eddc3492d30` 下载了含工程和 MDD 的 ZIP，校验通过。全套 756 项测试通过，零跳过。测试中还实际触发过一次插件退出失败，系统记录原因并完成内置头发回退。

证据见 `validation/mmd-connector-dev28/release/` 下的 run.json、export.json、download.json 和 Rin-MMD-Blender.zip，以及上一级目录的 final-suite.xml。物理前后工程及独立回读报告在 run.json 的 result.pair 指向的目录。
