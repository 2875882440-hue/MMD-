# MotionBridge

面向 Blender 动画创作者的 Windows 本地动捕工作台。导入舞蹈视频后，可以预览骨架、导出动作文件，并将动作转到已配置的角色。远坂凛另有可选的 MMD Tools 衣发处理入口。

## 启动基础工作台

1. 安装 Windows 版 Python 3.12（含 py 启动器）。在项目根目录双击「安装或修复.bat」，等待 Python 依赖和 MediaPipe 模型下载完成。
2. 复制 config.example.json 为 config.json，按本机安装位置填写 blender_path。需要导出角色时再准备对应资产。
3. 双击「启动 MotionBridge.bat」，浏览器打开 http://127.0.0.1:8766/ 。先上传一个短视频，查看骨架预览，再选择需要的导出格式。

## 高级功能与外部资产

GVHMR 需要独立的 CUDA 环境、上游代码、权重和经授权的人体模型，本版单次上限 32 秒。MMD Tools 通道验证使用便携 Blender 5.2.4 和 MMD Tools 4.5.14，独立于主 Blender。请自行取得所需模型和软件，按角色配置提供匹配的 PMX、工程和蒙皮。仓库包含适配代码和配置，模型实体、个人视频和任务缓存保存在本机。

使用步骤见 [Blender 插件说明](docs/BLENDER_PLUGIN_BRIDGE.md)。导出插件结果时请整包解压，工程需要同包的 MDD 缓存才能播放衣发运动。

## 首次源码发布

版本为 **0.7.0-dev.28**。开发环境完成过 756 项测试，并验证过凛工程、衣发缓存及 Blender 导出。部分回归需要开发机的模型和历史素材。裙片限幅候选改善了个别帧，但整段穿插计数回升，未纳入本版。详见[复核记录](docs/RIN_MMD_SKIRT_KNEE_SWING_REVIEW.md)、[版本日志](CHANGELOG.md)和[第三方组件说明](THIRD_PARTY_NOTICES.md)。历史报告里的盘符及任务编号用于定位开发机上的验证记录。
