# 正常子群实现核验与零更新预检

当前入口只支持合成核验和原模型零更新冷重放，没有正式训练模式。不得用R30训练脚本代跑。最新规格为`configs/normal-subgroup-refinement-20261118/design.json`，`execution_enabled=false`。

## 合成核验

使用已有兼容环境；Python 3.12、torch 2.6.0+cu126、tabm 0.0.3、rtdl_num_embeddings 0.0.12及原NumPy/pandas/sklearn依赖，不升级包。

```powershell
.\scripts\CHECK_NORMAL_SUBGROUP_DESIGN.ps1 -CheckOnly -Python 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe'
```

这一步只使用合成数组与新建微型官方TabM。验证分组/分位目标、k32损失与梯度、双前向双反传、固定轮次选择、冷重放接口；不读取真实权重或对真实流量训练。

本机核验复用了仓库现有`local/model-training-test-deps`中的固定作者包，通过命令级`PYTHONPATH`引入；未安装或升级系统依赖。

## 零更新预检入口

后续实验室预检可以使用下列命令。`R30Archive`填写已解压的R30回传根目录，该目录应直接含`inputs`、`models`、`freeze.json`。输出须为新的空目录。

```powershell
.\scripts\CHECK_NORMAL_SUBGROUP_DESIGN.ps1 -ZeroUpdate `
  -Python 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe' `
  -R30Archive '实际R30回传根目录' `
  -OutputDir '本项目下新的零更新预检输出目录'
```

代码固定CPU单线程，依次对三个blind原工件各冷加载两次。只读取F/V，逐位比较两次同路径输出，核对权重文件未被改写；再生成正式同路径F目标和V参照。相同模型输入若仍有不同参照分数，直接拒绝，不修改identity目标或放宽容差。不创建优化器、不执行更新、不读取C/E、不校准阈值。

输出包括：输入SHA收据、冻结分组、F/V逐行身份、每seed原分数、目标数组、目标元数据、预处理、输出SHA清单和状态文件。成功状态是`zero_update_preflight_complete`，仍为`training_ready=false`。模型重放阶段失败写入状态文件；运行前环境或输入校验失败直接报错，不创建结果目录。非空目录拒绝自动重复运行。

本次开发已核验该入口的合成后端、序列化与拒绝条件，尚未运行真实权重冷重放。旧GPU F缓存上的目标诊断不能替代本入口的CPU参照。后续训练runner、完整六工件冻结和一次C/E评价接线完成后，再生成独立正式训练注册；本入口不会自动进入该阶段。
