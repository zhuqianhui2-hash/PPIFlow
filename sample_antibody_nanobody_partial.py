"""
Backbone Structure Sampling Pipeline - Binder Design (Optimized for Chain Identification)
"""

import os
import random
import re
import dataclasses
import yaml
import time
import numpy as np
import pandas as pd
from Bio import PDB
from typing import Dict, Any, List, Optional, Set
import argparse
from omegaconf import OmegaConf

from experiments.inference_antibody_partial import Experiment
from data import utils as du
from data import parsers
from data import errors

# =============================================================================
# Index & Mask Utilities (完善的链识别逻辑)
# =============================================================================

def get_clean_res_list(structure, requested_chains: List[str]):
    """
    根据用户指定的链ID顺序，从Biopython结构中提取残基列表。
    确保最终生成的特征数组顺序与这里的残基顺序完全一致。
    """
    ordered_residues = []
    # 建立链映射，方便快速查找
    chain_map = {c.id: c for c in structure.get_chains()}
    
    for cid in requested_chains:
        if cid not in chain_map:
            raise ValueError(f"Chain {cid} not found in PDB file!")
        
        chain = chain_map[cid]
        # 只保留标准氨基酸
        for res in chain:
            if PDB.is_aa(res, standard=True):
                ordered_residues.append(res)
    return ordered_residues

def generate_masks_by_id(ordered_residues, antigen_chains: List[str], 
                         fixed_spec: str, cdr_spec: str):
    """
    核心逻辑：不再使用切片，而是遍历残基列表，根据每个残基所属的链和编号打标签。
    """
    total_len = len(ordered_residues)
    fix_mask = np.zeros(total_len, dtype=np.int8)
    cdr_mask = np.zeros(total_len, dtype=np.int8)

    # 预处理用户输入的规格字符串
    fixed_targets = set(expand_ranges(fixed_spec).split(",")) if fixed_spec else set()
    cdr_targets = set(expand_ranges(cdr_spec).split(",")) if cdr_spec else set()

    for i, res in enumerate(ordered_residues):
        chain_id = res.get_parent().id
        res_num = str(res.get_id()[1])
        icode = res.get_id()[2].strip()
        full_res_id = f"{chain_id}{res_num}{icode}"

        # 1. 抗原自动全部固定
        if chain_id in antigen_chains:
            fix_mask[i] = 1
        
        # 2. 如果在用户指定的 fixed_positions 列表中，也固定
        if full_res_id in fixed_targets:
            fix_mask[i] = 1
            
        # 3. 如果在 CDR 列表中，标记为 CDR (用于采样引导)
        if full_res_id in cdr_targets:
            cdr_mask[i] = 1

    return fix_mask, cdr_mask

def expand_ranges(s: str) -> str:
    """展开 H26-33 这种格式"""
    if not s or "-" not in s:
        return s or ""
    result = []
    parts = s.split(",")
    for part in parts:
        part = part.strip()
        match = re.match(r"([A-Za-z]+)(\d+)-(\d+)", part)
        if match:
            prefix, start, end = match.groups()
            for i in range(int(start), int(end) + 1):
                result.append(f"{prefix}{i}")
        else:
            result.append(part)
    return ",".join(result)

# =============================================================================
# Preprocessing Logic
# =============================================================================

def process_file_robust(input_info, write_dir, sample_id):
    pdb_name = input_info["pdb_name"]
    filepath = input_info["complex_pdb"]
    
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("tmp", filepath)

    # 1. 确定我们要处理的链及其顺序 (抗体在前，抗原在后，这个顺序决定了特征数组的排列)
    ab_chains = [input_info["heavy_chain"]]
    if input_info["light_chain"]:
        ab_chains.append(input_info["light_chain"])
    
    ag_chains = list(input_info["antigen_chain"])  # 比如 'E'
    
    # 这里的顺序决定了模型的“视野”
    all_requested_chains = ab_chains + ag_chains 

    # 2. 按照这个顺序提取残基
    ordered_residues = get_clean_res_list(structure, all_requested_chains)
    
    # 3. 精确生成掩码，基于每个残基真实的 Chain ID 和 Position
    fix_mask, cdr_mask = generate_masks_by_id(
        ordered_residues, 
        antigen_chains=ag_chains,
        fixed_spec=input_info["fixed_positions"],
        cdr_spec=input_info["cdr_position"]
    )

    # 4. 提取特征 (同样按照 all_requested_chains 顺序)
    struct_map = {c.id: c for c in structure.get_chains()}
    struct_feats = []
    for cid in all_requested_chains:
        chain_obj = struct_map[cid]
        chain_id_int = du.chain_str_to_int(cid)
        chain_prot = parsers.process_chain(chain_obj, chain_id_int)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = du.parse_chain_feats(chain_dict, normalize_positions=False)
        struct_feats.append(chain_dict)
    
    complex_feats = du.concat_np_features(struct_feats, False)

    # 5. 设置 chain_groups (0=Antigen, 1=Antibody)
    ag_chain_ints = [du.chain_str_to_int(c) for c in ag_chains]
    ab_chain_ints = [du.chain_str_to_int(c) for c in ab_chains]
    
    complex_feats["chain_groups"] = np.where(
        np.isin(complex_feats["chain_index"], ag_chain_ints), 0, 1
    )
    complex_feats["fix_structure_mask"] = fix_mask
    complex_feats["cdr_mask"] = cdr_mask

    # 6. 处理 Hotspots
    # 使用修改后的逻辑获取热点在数组中的索引
    hotspot_indices = []
    hotspot_targets = set(expand_ranges(input_info["hotspots"]).split(","))
    for i, res in enumerate(ordered_residues):
        cid = res.get_parent().id
        rid = f"{cid}{res.get_id()[1]}{res.get_id()[2].strip()}"
        if rid in hotspot_targets:
            hotspot_indices.append(i)

    # 7. 保存与元数据
    save_id = f"{pdb_name}_sample_{sample_id}"
    processed_path = os.path.join(write_dir, f"{save_id}.pkl")
    du.write_pkl(processed_path, complex_feats)

    metadata = {
        "id": save_id,
        "pdb_name": save_id,
        "target_id": ag_chains,
        "binder_id": ab_chains,
        "processed_path": processed_path,
        "hotspots": hotspot_indices,
        "seq_len": len(ordered_residues)
    }
    return metadata

# =============================================================================
# Main Pipeline Wrapper
# =============================================================================

def run_pipeline(args):
    print(f">>> Loading structure from: {args.pf_complex_pdb}")
    output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)
    input_data_dir = os.path.join(output_dir, "input")
    os.makedirs(input_data_dir, exist_ok=True)

    input_info = {
        "antigen_chain": args.pf_antigen_chain,
        "heavy_chain": args.pf_heavy_chain,
        "light_chain": args.pf_light_chain,
        "pdb_name": "task",
        "hotspots": args.pf_specified_hotspots,
        "fixed_positions": args.pf_fixed_positions,
        "cdr_position": args.pf_cdr_position,
        "complex_pdb": args.pf_complex_pdb,
    }

    # 预处理数据
    csv_path = os.path.join(input_data_dir, "input_manifest.csv")
    metadatas = []
    for i in range(args.pf_samples_per_target):
        meta = process_file_robust(input_info, input_data_dir, i)
        metadatas.append(meta)
    pd.DataFrame(metadatas).to_csv(csv_path, index=False)

    # 配置更新
    conf_mgr = ConfigManager(args.config)
    updates = {
        "ppi_dataset": {"test_csv_path": csv_path, "samples_per_target": 1},
        "experiment": {
            "testing_model": {"ckpt_path": args.model_weights, "save_dir": output_dir},
            "retry_Limit": args.pf_retry_limit
        },
        "interpolant": {"min_t": args.pf_start_t}
    }
    conf_mgr.update_config(updates)
    
    final_conf_path = os.path.join(output_dir, "run_config.yaml")
    conf_mgr.save_config(final_conf_path)

    # 运行模型
    print(">>> Starting Inference...")
    cfg = OmegaConf.load(final_conf_path)
    exp = Experiment(cfg=cfg)
    exp.test()

class ConfigManager:
    def __init__(self, path):
        with open(path, 'r') as f: self.config = yaml.safe_load(f)
    def update_config(self, updates):
        def _update(d, u):
            for k, v in u.items():
                if isinstance(v, dict): d[k] = _update(d.get(k, {}), v)
                else: d[k] = v
            return d
        self.config = _update(self.config, updates)
    def save_config(self, path):
        with open(path, 'w') as f: yaml.dump(self.config, f)

# =============================================================================
# CLI Parser
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 对应你输入的 CLI 参数名
    parser.add_argument("--pf-complex-pdb", type=str, required=True)
    parser.add_argument("--pf-antigen-chain", type=str, required=True)
    parser.add_argument("--pf-heavy-chain", type=str, required=True)
    parser.add_argument("--pf-light-chain", type=str, default=None)
    parser.add_argument("--pf-fixed-positions", type=str, default="")
    parser.add_argument("--pf-cdr-position", type=str, default="")
    parser.add_argument("--pf-specified-hotspots", type=str, default="")
    parser.add_argument("--pf-start-t", type=float, default=0.2)
    parser.add_argument("--pf-samples-per-target", type=int, default=1)
    parser.add_argument("--pf-retry-limit", type=int, default=10)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-weights", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    # 忽略 modal run 的其他参数
    args, unknown = parser.parse_known_args()

    run_pipeline(args)
