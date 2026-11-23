# 第三方组件与来源

本文件记录 v0.1.0 的主要运行依赖，具体安装版本见 requirements-lock.txt。各依赖原有许可保留在其安装包中；此文件不替代原始许可。

- MediaPipe / Pose Landmarker: https://github.com/google-ai-edge/mediapipe ; Apache-2.0 代码。模型下载自 Google 官方 mediapipe-models 存储桶，固定路径版本 1。模型说明：https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker 。下载校验记录见 models/manifest.json。
- Three.js r160: https://github.com/mrdoob/three.js/tree/r160 ; MIT。项目已保留 `motionbridge/static/vendor/THREE-LICENSE.txt`。
- FastAPI: https://github.com/fastapi/fastapi ; MIT。
- Uvicorn: https://github.com/encode/uvicorn ; BSD-3-Clause。
- OpenCV: https://github.com/opencv/opencv ; Apache-2.0，包含第三方组件许可。
- NumPy: https://github.com/numpy/numpy ; BSD-3-Clause。
- SciPy: https://github.com/scipy/scipy ; BSD-3-Clause。
- imageio-ffmpeg: https://github.com/imageio/imageio-ffmpeg ; BSD-2-Clause wrapper，捆绑 FFmpeg 二进制具有自己的许可，分发产品时需一并核查。
- Blender: https://www.blender.org/ ; 本软件调用用户已安装的 Blender，未将 Blender 二进制打包。

彩色骨架按 BODY18 关节顺序和常用 OpenPose 渲染约定编写，本项目没有集成或分发 CMU OpenPose 引擎。姿态模型使用 MediaPipe，不声称具备 DWPose 的完整手部/面部检测。

验证用外部视频的来源与用途记在 docs/VALIDATION.md；它们用于开发验证，不作为可商用素材授权。

v0.4.0 新增可选 MediaPipe Hand Landmarker 官方版本 1，用于局部手腕位置辅助，不输出手指动画。来源为 `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`；长度 7,819,105 字节、官方对象 MD5、SHA-256 和 ZIP CRC 校验记录见 `models/hand_manifest.json`。模型和依赖沿用其原始许可，不包含对用户角色素材的商用授权。


## Secondary-motion PMX inspection (development)
Bundled unchanged PMX parser from locally installed MMD Tools v4.5.14, GPL-3.0-or-later. Source: https://github.com/MMD-Blender/blender_mmd_tools . License retained at motionbridge/vendor/MMD_TOOLS_LICENSE.txt. Model assets retain their separate author rules and are for local research, not redistribution.
