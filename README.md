# 2025-Huawei-Intelligent-Link-Cup
We won the second prize in the Unlimited Programming Competition of this event and ranked within the top ten in the preliminary round. Although the code is not perfect, we hope it can provide some ideas and references for those working on related topics

---

### 总体架构
赛题一的模型(model文件夹）对硬件平台的任务指标向量进行功耗预测
赛题二的代码实现了一个面向异构计算平台的任务调度系统，核心目标是在满足硬件资源约束和依赖关系的前提下，最小化计算图的整体推理时延。系统采用**动态优先级调度**与**内存感知的资源管理**策略，结合拓扑排序与贪心算法进行任务分配。

---

### 模块一：`Node` 类
**作用**：表示计算图中的子图节点（最小执行单元）。  
**关键属性**：
- `op_id`, `node_id`, `tiling`：标识节点所属算子、节点ID及采用的Tiling策略。
- `core_type`, `exec_time`：执行所需的核类型与耗时。
- `mem_reqs_idx`, `mem_reqs_size`：内存类型索引及对应大小，通过 `mem_map` 字典聚合。
- `pre_count`, `succs`, `preds`：维护节点依赖关系。
- `mem_succ_counters`：跟踪后续节点对内存的依赖情况，支持延迟释放机制。

**设计意义**：封装节点的所有调度相关信息，为依赖管理和资源分配提供基础数据结构。

---

### 模块二：`SystemResource` 类
**作用**：管理硬件资源（核与内存）的分配与释放。  
**核心方法**：
- `set_cores()` / `set_memory()`：初始化核和内存的资源池。
- `can_allocate()`：检查当前资源是否满足节点的核与内存需求（支持预释放内存评估）。
- `allocate()` / `release_core()` / `release_memory()`：执行资源的分配与释放操作。

**设计意义**：解耦资源管理与调度逻辑，确保资源操作的一致性与可维护性。

---

### 模块三：`TaskScheduler` 类
**作用**：调度器核心，负责输入解析、图构建、调度策略执行与结果生成。  

#### 1. **输入解析（`parse_line()`）**
- 使用正则表达式匹配输入行，解析硬件平台信息（`SetSocInfo`）、算子信息（`AddOpInfo`）和计算图结构（`GetInferenceScheResult`）。
- 将解析后的数据存储于 `variants`、`ops`、`op_deps` 等字典中。

#### 2. **Tiling预处理（`_dp_preprocess_tiling()`）**
- 对每个算子的Tiling策略进行动态规划预处理，计算其执行所需的最小峰值内存。
- 通过拓扑排序遍历内部子图节点，聚合内存需求，排除内存超限的Tiling选项。

#### 3. **节点图构建（`build_nodes()`）**
- 为每个算子选择最优Tiling策略（基于内存使用与执行时间的加权评估）。
- 构建全局节点图，跨算子连接依赖关系（算子间依赖转换为子图节点间的边）。
- 初始化节点的内存依赖计数器（`mem_succ_counters`），为延迟释放机制提供依据。

#### 4. **调度执行（`schedule()`）**
- 使用优先队列（`finish_heap`）管理节点完成事件，`ready_nodes` 维护可调度节点。
- **调度策略**：
  - **优先级**：优先调度内存需求大、算子ID小的节点。
  - **内存准入检查**：在调度算子首个节点时，检查系统内存（含预释放内存）是否满足该算子的峰值内存需求。
  - **延迟内存释放**：节点完成后，仅当其内存未被后续节点依赖时才立即释放，否则延迟至最后依赖节点启动时释放。
  - **死锁处理**：当无节点可调度时，强制释放已完成节点的被依赖内存以打破僵局。
- 输出每个节点的调度信息：`[op_id, tiling, node_id, start_time, core_id]`。

---

### 模块四：`main` 函数
**作用**：程序入口，读取输入数据，调用调度器并输出结果。  
**流程**：
1. 逐行读取输入，交由 `TaskScheduler` 解析。
2. 调用 `build_nodes()` 构建节点图。
3. 执行 `schedule()` 生成调度方案。
4. 按开始时间排序后输出结果。
