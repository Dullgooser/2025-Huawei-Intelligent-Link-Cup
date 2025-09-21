import sys
import re
import heapq
import ast
import numpy as np
from collections import defaultdict

class Node:
    """
    增强的节点类,增加了支持延迟内存释放和依赖追踪的属性
    """
    __slots__ = ('op_id', 'tiling', 'node_id', 'core_type', 'exec_time',
                 'mem_reqs_idx', 'mem_reqs_size', 'pre_count', 'succs', 'preds',
                 'ready_time', 'is_finished', 'mem_succ_counters', 'mem_map', 'total_mem_req')

    def __init__(self, op_id, tiling, node_id, core_type, exec_time, mem_reqs_idx, mem_reqs_size):
        self.op_id = op_id
        self.tiling = tiling
        self.node_id = node_id
        self.core_type = core_type
        self.exec_time = exec_time
        self.mem_reqs_idx = mem_reqs_idx
        self.mem_reqs_size = mem_reqs_size
        self.mem_map = {idx: size for idx, size in zip(mem_reqs_idx, mem_reqs_size)}
        self.total_mem_req = np.sum(mem_reqs_size)
        self.pre_count = 0
        self.succs = []
        self.preds = []
        self.ready_time = 0
        self.is_finished = False
        self.mem_succ_counters = defaultdict(int)

class SystemResource:
    """
    资源管理类,分离了核心和内存的释放逻辑
    """
    __slots__ = ('core_info', 'mem_index', 'mem_array', 'mem_count', 'initial_mem_array')

    def __init__(self):
        self.core_info = {}
        self.mem_index = {}
        self.mem_array = np.array([], dtype=np.int64)
        self.initial_mem_array = np.array([], dtype=np.int64) 
        self.mem_count = 0

    def set_cores(self, core_data):
        for ctype, num in core_data:
            heap = [(0, idx) for idx in range(num)]
            heapq.heapify(heap)
            self.core_info[ctype] = heap

    def set_memory(self, mem_data):
        if not mem_data: return
        types, sizes = zip(*[(mtype, int(size)) for mtype, size in mem_data])
        self.mem_index = {mtype: idx for idx, mtype in enumerate(types)}
        self.mem_array = np.array(sizes, dtype=np.int64)
        self.initial_mem_array = np.copy(self.mem_array)
        self.mem_count = len(self.mem_index)

    def can_allocate(self, core_type, mem_reqs_idx, mem_reqs_size, extra_mem=None):
        heap = self.core_info.get(core_type)
        if not heap: return False
        current_mem = self.mem_array
        if extra_mem is not None: current_mem = current_mem + extra_mem
        if mem_reqs_idx.size > 0:
            if np.any(current_mem[mem_reqs_idx] < mem_reqs_size): return False
        return True

    def allocate(self, core_type, mem_reqs_idx, mem_reqs_size):
        free_time, core_idx = heapq.heappop(self.core_info[core_type])
        if mem_reqs_idx.size > 0: np.subtract.at(self.mem_array, mem_reqs_idx, mem_reqs_size)
        return free_time, core_idx

    def release_core(self, core_type, core_idx, finish_time):
        heapq.heappush(self.core_info[core_type], (finish_time, core_idx))

    def release_memory(self, mem_to_release):
        if not mem_to_release: return
        indices, sizes = zip(*mem_to_release.items())
        np.add.at(self.mem_array, np.array(indices), np.array(sizes))

class TaskScheduler:
    __slots__ = ('res', 'variants', 'ops', 'op_deps', 'nodes', 'nodes_by_op', 're_set', 're_add', 're_get', 'preprocessed_tilings', 'op_peak_info')

    def __init__(self):
        self.res = SystemResource()
        self.variants = defaultdict(dict)
        self.ops = {}
        self.op_deps = defaultdict(list)
        self.nodes = {}
        self.nodes_by_op = defaultdict(list)
        self.re_set = re.compile(r"SetSocInfo\((\[\[.*\]\]),\s*(\[\[.*\]\])\)")
        self.re_add = re.compile(r"AddOpInfo\((.*)\)")
        self.re_get = re.compile(r"GetInferenceScheResult\((.*)\)")
        self.preprocessed_tilings = {}
        self.op_peak_info = {} # 新增：用于缓存每个算子选定Tiling后的峰值信息

    def parse_line(self, line):
        line = line.strip()
        if not line: return
        m = self.re_set.match(line)
        if m:
            self.res.set_cores(ast.literal_eval(m.group(1))); self.res.set_memory(ast.literal_eval(m.group(2)))
            return
        m = self.re_add.match(line)
        if m:
            op_type, shape, tiling, deps, runs, mems = ast.literal_eval(f"[{m.group(1)}]")
            self.variants[(op_type, shape)][tiling] = {'deps': deps, 'runs': runs, 'mem': mems}
            return
        m = self.re_get.match(line)
        if m:
            edges, nodes_info = ast.literal_eval(f"[{m.group(1)}]")
            for u, v in edges: self.op_deps[u].append(v)
            for op_id, op_type, shape in nodes_info: self.ops[op_id] = (op_type, shape)

    def _dp_preprocess_tiling(self, runs, deps, mems):
        if not runs: return np.zeros(self.res.mem_count, dtype=np.int64)
        internal_nodes = {r[0]: {'succs': [], 'pred_count': 0} for r in runs}
        node_mem_reqs = defaultdict(lambda: np.zeros(self.res.mem_count, dtype=np.int64))
        for nid, mtype, sz in mems:
            if nid in internal_nodes:
                mem_idx = self.res.mem_index.get(mtype)
                if mem_idx is not None: node_mem_reqs[nid][mem_idx] += sz
        for u, v in deps:
            if u in internal_nodes and v in internal_nodes:
                internal_nodes[u]['succs'].append(v); internal_nodes[v]['pred_count'] += 1
        queue = [nid for nid, data in internal_nodes.items() if data['pred_count'] == 0]
        topo_order, head = [], 0
        while head < len(queue):
            u = queue[head]; head += 1; topo_order.append(u)
            for v in internal_nodes[u]['succs']:
                internal_nodes[v]['pred_count'] -= 1
                if internal_nodes[v]['pred_count'] == 0: queue.append(v)
        if len(topo_order) != len(internal_nodes): return np.full(self.res.mem_count, np.iinfo(np.int64).max)
        dp_mem_vals = {}
        for node_id in reversed(topo_order):
            max_succ_mem = np.zeros(self.res.mem_count, dtype=np.int64)
            if internal_nodes[node_id]['succs']:
                succ_mems = np.array([dp_mem_vals[succ_id] for succ_id in internal_nodes[node_id]['succs']])
                if succ_mems.size > 0: max_succ_mem = np.max(succ_mems, axis=0)
            dp_mem_vals[node_id] = node_mem_reqs[node_id] + max_succ_mem
        root_nodes = [nid for nid, data in internal_nodes.items() if data['pred_count'] == 0]
        final_peak_mem = np.zeros(self.res.mem_count, dtype=np.int64)
        if root_nodes:
             root_mems = np.array([dp_mem_vals[root_id] for root_id in root_nodes])
             if root_mems.size > 0: final_peak_mem = np.max(root_mems, axis=0)
        return final_peak_mem

    def build_nodes(self):
        for (op_type, shape), variants in self.variants.items():
            for tiling_id, info in variants.items():
                min_mem_req = self._dp_preprocess_tiling(info['runs'], info['deps'], info['mem'])
                self.preprocessed_tilings[(op_type, shape, tiling_id)] = {'min_mem_req': min_mem_req}
        chosen_tilings = {}
        for op_id, (op_type, shape) in self.ops.items():
            variants_for_op = self.variants.get((op_type, shape), {})
            feasible_options = []
            for tiling_id, info in variants_for_op.items():
                proc_info = self.preprocessed_tilings[(op_type, shape, tiling_id)]
                if self.res.mem_count > 0 and np.any(proc_info['min_mem_req'] > self.res.initial_mem_array): continue
                
                total_mem_usage = sum(m[2] for m in info['mem'])
                total_runtime = sum(r[2] for r in info['runs'])
                score = (total_mem_usage, total_runtime) 
                feasible_options.append((score, tiling_id))

            if feasible_options:
                _, best_tiling = min(feasible_options)
                chosen_tilings[op_id] = best_tiling
            elif variants_for_op:
                chosen_tilings[op_id] = next(iter(variants_for_op))

        for op_id, tiling_id in chosen_tilings.items():
            op_type, shape = self.ops[op_id]
            proc_info = self.preprocessed_tilings.get((op_type, shape, tiling_id))
            if proc_info:
                self.op_peak_info[op_id] = proc_info['min_mem_req']
            else:
                self.op_peak_info[op_id] = np.zeros(self.res.mem_count, dtype=np.int64)
        
        for op_id, tiling_id in chosen_tilings.items():
            op_type, shape = self.ops[op_id]
            info = self.variants[(op_type, shape)][tiling_id]
            runs, deps, mems = info['runs'], info['deps'], info['mem']
            mem_map = defaultdict(lambda: ([], []))
            for nid, mtype, sz in mems:
                idx = self.res.mem_index.get(mtype)
                if idx is not None: mem_map[nid][0].append(idx); mem_map[nid][1].append(int(sz))
            for nid, ctype, dur in runs:
                idx_list, size_list = mem_map.get(nid, ([], []))
                node = Node(op_id, tiling_id, nid, ctype, dur, np.array(idx_list, dtype=np.int64), np.array(size_list, dtype=np.int64))
                self.nodes[(op_id, nid)] = node; self.nodes_by_op[op_id].append(node)
            for src, dst in deps:
                s, d = self.nodes.get((op_id, src)), self.nodes.get((op_id, dst))
                if s and d: s.succs.append(d); d.pre_count += 1
        op_initials, op_terminals = defaultdict(list), defaultdict(list)
        for op_id, op_nodes in self.nodes_by_op.items():
            for node in op_nodes:
                if node.pre_count == 0: op_initials[op_id].append(node)
                if not any(s.op_id == op_id for s in node.succs): op_terminals[op_id].append(node)
        for u, vs in self.op_deps.items():
            for v in vs:
                for src_node in op_terminals.get(u, []):
                    for dst_node in op_initials.get(v, []):
                        if dst_node not in src_node.succs: src_node.succs.append(dst_node); dst_node.pre_count += 1
        node_mem_map = {n: set(n.mem_reqs_idx) for n in self.nodes.values()}
        for s in self.nodes.values():
            s_mem_types = node_mem_map[s]
            for d in s.succs:
                d.preds.append(s)
                for mem_idx in s_mem_types.intersection(node_mem_map[d]):
                    s.mem_succ_counters[mem_idx] += 1

    def schedule(self):
        res = self.res
        finish_heap = []
        result = []
        tie_breaker = 0
        curr_time = 0
        ready_nodes = [node for node in self.nodes.values() if node.pre_count == 0]
        
        started_ops = set()

        while ready_nodes or finish_heap:
            was_progress_made = True
            while was_progress_made:
                was_progress_made = False
                candidates = sorted([n for n in ready_nodes if n.ready_time <= curr_time], key=lambda n: (-n.total_mem_req, n.op_id, n.node_id))
                if not candidates: break
                
                for node in candidates:
                    # --- 核心改动：将“可释放内存”的计算提前 ---
                    # 无论后续检查是否需要，都先计算好调度此节点能带来的内存增益
                    mem_to_release = defaultdict(np.int64)
                    node_mem_types = set(node.mem_reqs_idx)
                    for pred in node.preds:
                        if pred.is_finished:
                            for mem_idx in node_mem_types:
                                if pred.mem_succ_counters.get(mem_idx, 0) == 1 and mem_idx in pred.mem_map:
                                    mem_to_release[mem_idx] += pred.mem_map[mem_idx]
                    temp_extra_mem = np.zeros(res.mem_count, dtype=np.int64)
                    if mem_to_release:
                        indices, sizes = zip(*mem_to_release.items())
                        temp_extra_mem[np.array(indices)] += np.array(sizes)

                    # --- 算子准入检查 ---
                    if node.op_id not in started_ops:
                        op_peak_array = self.op_peak_info[node.op_id]
                        # --- 核心改动：在检查时，加上“可释放内存” ---
                        # 新的、更智能的检查逻辑：
                        # 用（当前可用内存 + 调度此节点能立刻释放的内存）来和理论峰值比较
                        if np.any((res.mem_array + temp_extra_mem) < op_peak_array):
                            continue # 如果依然不够，则搁置
                    
                    if res.can_allocate(node.core_type, node.mem_reqs_idx, node.mem_reqs_size, extra_mem=temp_extra_mem):
                        res.release_memory({idx: size for idx, size in mem_to_release.items() if size > 0})
                        core_free_time, core_idx = res.allocate(node.core_type, node.mem_reqs_idx, node.mem_reqs_size)
                        start_time = max(curr_time, core_free_time)
                        finish_time = start_time + node.exec_time
                        heapq.heappush(finish_heap, (finish_time, tie_breaker, node, core_idx)); tie_breaker += 1
                        result.append([node.op_id, node.tiling, node.node_id, start_time, core_idx])
                        
                        started_ops.add(node.op_id)
                        ready_nodes.remove(node)
                        for pred in node.preds:
                            if pred.is_finished:
                                for mem_idx in node_mem_types.intersection(pred.mem_map.keys()):
                                    if mem_idx in pred.mem_succ_counters: pred.mem_succ_counters[mem_idx] -= 1
                        
                        was_progress_made = True
                        break
            
            if not ready_nodes and not finish_heap: break
            
            if not finish_heap:
                finished_holding_mem = [n for n in self.nodes.values() if n.is_finished and any(n.mem_succ_counters.get(idx, 0) > 0 for idx in n.mem_map)]
                if not finished_holding_mem:
                    break 
                
                victim = max(finished_holding_mem, key=lambda n: n.total_mem_req)
                mem_to_force_release = {idx: victim.mem_map[idx] for idx, count in victim.mem_succ_counters.items() if count > 0}
                res.release_memory(mem_to_force_release)
                victim.mem_succ_counters.clear()
                continue

            next_finish_time = finish_heap[0][0]
            curr_time = max(curr_time, next_finish_time)
            
            while finish_heap and finish_heap[0][0] <= curr_time:
                fin_time, _, node_f, core_idx = heapq.heappop(finish_heap)
                node_f.is_finished = True
                res.release_core(node_f.core_type, core_idx, fin_time)
                for mem_idx, mem_size in node_f.mem_map.items():
                    if node_f.mem_succ_counters.get(mem_idx, 0) == 0:
                        res.release_memory({mem_idx: mem_size})
                for child_node in node_f.succs:
                    child_node.pre_count -= 1
                    child_node.ready_time = max(child_node.ready_time, fin_time)
                    if child_node.pre_count == 0: ready_nodes.append(child_node)
        return result

def main():
    sched = TaskScheduler()
    lines = []
    while True:
        try:
            l = input().strip()
            if not l: break
            lines.append(l)
        except EOFError: break
    for l in lines: sched.parse_line(l)
    sched.build_nodes()
    out = sched.schedule()
    out.sort(key=lambda x: (x[3], x[0], x[2]))
    print(out)

if __name__ == '__main__':
    main()