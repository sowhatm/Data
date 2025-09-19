import pandas as pd
from pymatgen.core import Structure
from pathlib import Path
import re
from collections import Counter

# 内置电负性数据（鲍林标度）
ELECTRONEGATIVITY = {
    'H': 2.20, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
    'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16,
    'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83,
    'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55,
    'Br': 2.96, 'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33, 'Nb': 1.60, 'Mo': 2.16,
    'Tc': 1.90, 'Ru': 2.20, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69, 'In': 1.78, 'Sn': 1.96,
    'Sb': 2.05, 'Te': 2.10, 'I': 2.66, 'Cs': 0.79, 'Ba': 0.89, 'La': 1.10, 'Ce': 1.12,
    'Pr': 1.13, 'Nd': 1.14, 'Pm': 1.13, 'Sm': 1.17, 'Eu': 1.20, 'Gd': 1.20, 'Tb': 1.22, 'Dy': 1.23,
    'Ho': 1.24, 'Er': 1.24, 'Tm': 1.25, 'Yb': 1.10, 'Lu': 1.27, 'Hf': 1.30, 'Ta': 1.50, 'W': 2.36,
    'Re': 1.90, 'Os': 2.20, 'Ir': 2.20, 'Pt': 2.28, 'Au': 2.54, 'Hg': 2.00, 'Tl': 1.62, 'Pb': 2.33,
    'Bi': 2.02, 'Po': 2.00, 'At': 2.20, 'Fr': 0.70, 'Ra': 0.90, 'Ac': 1.10, 'Th': 1.30,
    'Pa': 1.50, 'U': 1.38
}


def parse_mixed_site(symbol):
    """解析混合位点，返回占据率最高的元素及其占据率"""
    elements = []
    parts = symbol.split(', ')
    for part in parts:
        match = re.match(r"([A-Za-z]+)\d*[+-]?:(\d*\.\d+)", part)
        if match:
            element, occupancy = match.groups()
            elements.append((element, float(occupancy)))
        else:
            match = re.match(r"([A-Za-z]+)\d*[+-]?", part)
            if match:
                elements.append((match.group(1), 1.0))

    if not elements:
        return None, 0.0
    dominant_element, dominant_occupancy = max(elements, key=lambda x: x[1])
    return dominant_element, dominant_occupancy


def get_electronegativity(symbol):
    """获取占据率最高元素的电负性"""
    dominant_element, _ = parse_mixed_site(symbol)
    if dominant_element:
        return ELECTRONEGATIVITY.get(dominant_element, None)
    return None


def clean_element_symbol(symbol):
    """从复杂符号中提取纯元素符号"""
    match = re.match(r"([A-Za-z]+)", symbol)
    return match.group(1) if match else symbol


def assign_roles_from_filename(filename):
    """从 CIF 文件名分配 A, B, M, X 角色"""
    elements = re.findall(r"([A-Z][a-z]?)(\d*)", filename.replace('.cif', ''))
    elem_count = Counter()
    for elem, count in elements:
        elem_count[elem] = int(count) if count else 1

    roles = {'A': None, 'B': None, 'M': None, 'X': None}
    element_list = [elem for elem, _ in elements]

    if element_list[-1] in ['O', 'F', 'Cl', 'Br', 'I']:
        roles['X'] = element_list[-1]
        element_list.pop(-1)
    if element_list:
        roles['A'] = element_list[0]
        element_list.pop(0)
    if element_list:
        roles['B'] = element_list[0]
        element_list.pop(0)
    if element_list:
        roles['M'] = element_list[0]

    return roles


def calculate_bond_lengths(structure, roles):
    """计算结构中不同原子间的最短键长，按照 A2BMX6 角色"""
    distances = structure.distance_matrix
    bond_lengths = {}
    role_map = {}

    # 映射原子到角色
    for site in structure.sites:
        atom = site.species_string
        if ',' in atom:  # 混合位点
            dominant_element, _ = parse_mixed_site(atom)
            if dominant_element and dominant_element in roles.values():
                matching_roles = [role for role, elem in roles.items() if elem == dominant_element]
                if matching_roles:
                    role_map[atom] = matching_roles[0]
        else:  # 纯元素位点
            elem = clean_element_symbol(atom)
            if elem in roles.values():
                matching_roles = [role for role, elem_roles in roles.items() if elem_roles == elem]
                if matching_roles:
                    role_map[atom] = matching_roles[0]

    # 计算键长
    for i, site1 in enumerate(structure.sites):
        for j, site2 in enumerate(structure.sites):
            if i < j:
                atom1 = site1.species_string
                atom2 = site2.species_string
                role1 = role_map.get(atom1)  # 使用 .get()，默认返回 None
                role2 = role_map.get(atom2)
                if role1 and role2 and role1 != role2:
                    distance = distances[i][j]
                    roles_pair = sorted([role1, role2])
                    bond_key = f"d({roles_pair[0]}-{roles_pair[1]})"
                    if bond_key not in bond_lengths or distance < bond_lengths[bond_key]:
                        bond_lengths[bond_key] = distance
    return bond_lengths


def process_cif_files(excel_path, cif_folder):
    """处理 Excel 文件中的 CIF 文件名并提取信息"""
    df = pd.read_excel(excel_path)
    cif_files = df['cif_filename'].tolist()

    results = []
    for cif_file in cif_files:
        if not cif_file.endswith('.cif'):
            cif_file = f"{cif_file}.cif"

        cif_path = Path(cif_folder) / cif_file
        if not cif_path.exists():
            print(f"文件 {cif_file} 不存在，跳过")
            continue

        try:
            structure = Structure.from_file(cif_path)
            roles = assign_roles_from_filename(cif_file)

            electronegativities = {}
            for site in structure.sites:
                atom = site.species_string
                en = get_electronegativity(atom)
                electronegativities[atom] = en

            bond_lengths = calculate_bond_lengths(structure, roles)

            result = {
                'filename': cif_file,
                'roles': roles,
                'electronegativities': electronegativities,
                'bond_lengths': bond_lengths
            }
            results.append(result)

            print(f"\n处理文件: {cif_file}")
            print("角色分配:", roles)
            print("电负性:")
            for atom, en in electronegativities.items():
                print(f"{atom}: {en:.2f}" if en is not None else f"{atom}: {en}")
            print("键长 (Å):")
            for bond, length in bond_lengths.items():
                print(f"{bond}: {length:.3f}")

        except Exception as e:
            print(f"处理 {cif_file} 时出错: {str(e)}")
            continue  # 继续处理下一个文件

    # 构建 DataFrame
    output_data = {'Filename': [r['filename'] for r in results]}
    for role in ['A', 'B', 'M', 'X']:
        output_data[f"X_{role}"] = []
        for r in results:
            role_elem = r['roles'][role]
            en = next((en for atom, en in r['electronegativities'].items() if
                       clean_element_symbol(atom) == role_elem or
                       (',' in atom and parse_mixed_site(atom)[0] == role_elem)), None)
            output_data[f"X_{role}"].append(en)

    bond_types = ['d(A-B)', 'd(A-M)', 'd(A-X)', 'd(B-M)', 'd(B-X)', 'd(M-X)']
    for bond in bond_types:
        output_data[bond] = [r['bond_lengths'].get(bond, None) for r in results]

    output_df = pd.DataFrame(output_data)
    output_df.to_excel('电负性键长特征.xlsx', index=False)
    print("\n结果已保存到 '电负性键长特征.xlsx'")


def main():
    excel_path = "cif.xlsx"
    cif_folder = "fold"
    Path(cif_folder).mkdir(exist_ok=True)
    process_cif_files(excel_path, cif_folder)


if __name__ == "__main__":
    main()