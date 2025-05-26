import json, hjson, json5, tomli, tomli_w, re
from ruamel.yaml import YAML
from lxml import etree
from zss import Node, simple_distance
import pandas as pd



def remove_jsonc_comments(text: str) -> str:
    def strip_line_comment(line):
        in_str = False
        esc = False
        result = ''
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == '"' and not esc:
                in_str = not in_str
            if not in_str and ch == '/' and i + 1 < len(line) and line[i + 1] == '/':
                break  # stop at //
            if not in_str and ch == '#' and (i == 0 or line[i - 1].isspace()):
                break  # treat # as YAML-style comment too
            result += ch
            esc = (ch == '\\') and not esc
            i += 1
        return result

    # 1. remove all /* ... */ comments (multi-line)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    # 2. remove // and # comments line by line
    lines = text.splitlines()
    lines = [strip_line_comment(line) for line in lines]
    return '\n'.join(lines)

def remove_yaml_comments(text: str) -> str:
    def strip_line_comment(line):
        in_str = False
        esc = False
        result = ''
        for i, ch in enumerate(line):
            if ch == '"' and not esc:
                in_str = not in_str
            if not in_str and ch == '#' and (i == 0 or line[i - 1].isspace()):
                break  # true comment
            result += ch
            esc = (ch == '\\') and not esc
        return result

    return '\n'.join(strip_line_comment(line) for line in text.splitlines())

def remove_toml_comments(text: str) -> str:
    def strip_line_comment(line):
        in_str = False
        esc = False
        result = ''
        for i, ch in enumerate(line):
            if ch == '"' and not esc:
                in_str = not in_str
            if not in_str and ch == '#' and (i == 0 or line[i - 1].isspace()):
                break
            result += ch
            esc = (ch == '\\') and not esc
        return result

    return '\n'.join(strip_line_comment(line) for line in text.splitlines())

def remove_xml_comments(text: str) -> str:
    return re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

def check_strict_syntax(text: str, fmt: str) -> int:
    try:
        if fmt == 'json':
            json.loads(text)
        elif fmt == 'jsonc':
            json.loads(remove_jsonc_comments(text))
        elif fmt == 'hjson':
            hjson.loads(remove_jsonc_comments(text))
        elif fmt == 'json5':
            json5.loads(remove_jsonc_comments(text))
        elif fmt == 'yaml':
            YAML().load(remove_yaml_comments(text))
        elif fmt == 'xml':
            etree.fromstring(remove_xml_comments(text).encode('utf-8'), parser=etree.XMLParser(recover=False))
        elif fmt == 'toml':
            tomli.loads(remove_toml_comments(text))
        else:
            return 0
        return 1
    except Exception:
        return 0
    
def loose_syntax(text: str, fmt: str) -> int:
    try:
        if fmt in ('json', 'jsonc', 'hjson','json5'):
            hjson.loads(remove_jsonc_comments(text))
        elif fmt == 'yaml':
            YAML().load(remove_yaml_comments(text))
        elif fmt == 'xml':
            etree.fromstring(remove_xml_comments(text).encode('utf-8'), parser=etree.XMLParser(recover=False))
        elif fmt == 'toml':
            tomli.loads(remove_toml_comments(text))
        else:
            return 0
        return 1
    except Exception:
        return 0

def soft_repair_input(text: str, fmt: str) -> str:
    """
    针对不同格式的文本内容进行软性修复，使其更易于解析。
    """
    if fmt in ('json', 'jsonc', 'hjson', 'json5'):
        repaired = re.sub(r'(?<=[0-9}\]"]) *(?=\")', ', ', text)
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
        return repaired

    elif fmt == 'yaml':
        # 修复冒号后无空格、行末多逗号（非法）
        repaired = re.sub(r':(?=\S)', ': ', text)                     # 修复 `key:value` → `key: value`
        repaired = re.sub(r',\s*\n', '\n', repaired)                 # 去掉行尾多逗号
        return repaired

    elif fmt == 'toml':
        repaired = ''
        for line in text.splitlines():
            if '=' not in line and re.match(r'^\w+\s*$', line):     # 形如 key，没有 =
                repaired += f'{line.strip()} = ""\n'
            else:
                repaired += line + '\n'
        return repaired

    elif fmt == 'xml':
        # 尝试修复一些未闭合符号（只用于 lxml 宽容解析前）
        repaired = re.sub(r'&(?!(amp|lt|gt|quot|apos);)', '&amp;', text)  # 替换未转义 &
        return repaired

    return text


def fix_text_loose(text: str, fmt: str) -> str:
    if fmt in ('json', 'jsonc', 'hjson','json5'):
        pre = soft_repair_input(text,fmt)
        return hjson.dumps(hjson.loads(pre), ensure_ascii=False, indent=2)
    if fmt == 'yaml':
        pre = soft_repair_input(text,fmt)
        yaml = YAML()
        data = yaml.load(pre)
        from io import StringIO
        buf = StringIO()
        yaml.dump(data, buf)
        return buf.getvalue()
    if fmt == 'xml':
        pre = soft_repair_input(text,fmt)
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(pre.encode('utf-8'), parser=parser)
        return etree.tostring(root, pretty_print=True, encoding='utf-8').decode('utf-8')
    if fmt == 'toml':
        pre = soft_repair_toml(text)
        data = tomli.loads(pre)
        return tomli_w.dumps(data)
    return text

def check_loose_syntax(text: str, fmt: str) -> int:
    try:
        fixed = fix_text_loose(text, fmt)
        return loose_syntax(fixed, fmt)
    except Exception:
        return 0


def parse_to_obj(text: str, fmt: str):
    try:
        if fmt in ('json', 'jsonc', 'hjson','json5'):
            cleaned = remove_jsonc_comments(text)
            return hjson.loads(soft_repair_input(cleaned, fmt))
        if fmt == 'yaml':
            cleaned = remove_yaml_comments(text)
            return YAML().load(soft_repair_input(cleaned, fmt))
        if fmt == 'xml':
            cleaned = remove_xml_comments(text)
            parser = etree.XMLParser(recover=True)
            return elem_to_dict(etree.fromstring(soft_repair_input(cleaned, fmt).encode('utf-8'), parser=parser))
        if fmt == 'toml':
            cleaned = remove_toml_comments(text)
            return tomli.loads(soft_repair_toml(cleaned))
        return {"__ERROR__": "Unsupported format"}
    except Exception as e:
        # ⛑ 出错时也返回一个可参与比较的结构
        return {"__ERROR__": f"parse failed: {str(e)}"}


def elem_to_dict(elem):
    d = {elem.tag: {} if elem.attrib else None}
    children = list(elem)
    if children:
        dd = {}
        for ch in children:
            cd = elem_to_dict(ch)
            for k, v in cd.items():
                dd.setdefault(k, []).append(v)
        d = {elem.tag: dd}
    if elem.text and elem.text.strip():
        text = elem.text.strip()
        if children or elem.attrib:
            d[elem.tag]['#text'] = text
        else:
            d[elem.tag] = text
    return d


def obj_to_zss(obj, label='root'):
    node = Node(str(label))
    
    # 如果是字典
    if isinstance(obj, dict):
        for k, v in obj.items():
            try:
                child = obj_to_zss(v, label=k)
            except Exception as e:
                child = Node(f"__ERROR__:{k}")
            node.addkid(child)
    
    # 如果是列表
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            try:
                child = obj_to_zss(item, label=f"[{i}]")
            except Exception:
                child = Node(f"__ERROR__:[{i}]")
            node.addkid(child)

    # 其他类型，作为值节点
    else:
        try:
            node.addkid(Node(str(obj)))
        except Exception as e:
            node.addkid(Node(f"__ERROR__:{label}"))

    return node



def count_nodes(node):
    return 1 + sum(count_nodes(c) for c in node.children)

def compute_ted_similarity(obj1, obj2):
    t1 = obj_to_zss(obj1, 'A')
    t2 = obj_to_zss(obj2, 'B')
    dist = simple_distance(t1, t2, get_children=lambda n: n.children, get_label=lambda n: n.label)
    size = max(count_nodes(t1), count_nodes(t2))
    norm = dist / size if size > 0 else 0
    sim = 1 - norm
    return dist, norm, sim

# ========== 测试 ==========

def soft_repair_toml(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' in line:
            # 原本就合法的等号语句，处理尾逗号和字符串引号
            line = re.sub(r',\s*$', '', line)  # 移除尾逗号
            parts = line.split('=', 1)
            if len(parts) == 2:
                key, val = parts[0].strip(), parts[1].strip()
                # 给没有引号的字符串加上引号（不影响数字、数组、true/false）
                if val and not re.match(r'^(\".*\"|\[.*\]|[0-9.\-]+|true|false)$', val, flags=re.I):
                    val = f'"{val}"'
                line = f"{key} = {val}"
        elif ':' in line:
            # 把冒号改为等号并处理值
            parts = line.split(':', 1)
            key, val = parts[0].strip(), parts[1].strip()
            val = re.sub(r',\s*$', '', val)  # 移除尾随逗号
            if val and not re.match(r'^(\".*\"|\[.*\]|[0-9.\-]+|true|false)$', val, flags=re.I):
                val = f'"{val}"'
            line = f"{key} = {val}"
        lines.append(line)
    return '\n'.join(lines)

def process_dataframe_with_validation(df: pd.DataFrame, gold_code: str):
    results = []

    for idx, row in df.iterrows():
        pid = row['participantId']
        fmt = row['format']
        code = row['code']

        # Strict and conditional loose check
        strict_result = check_strict_syntax(code, fmt)
        if strict_result == 1:
            loose_result = None
        else:
            loose_result = check_loose_syntax(code, fmt)

        # Determine final check
        final_check = 1 if strict_result == 1 or loose_result == 1 else 0

        # Parse both codes
        try:
            gold_obj = parse_to_obj(gold_code, fmt)
        except:
            gold_obj = {"__ERROR__": "parse failed"}
        try:
            part_obj = parse_to_obj(code, fmt) if final_check == 1 else {"__ERROR__": "invalid input"}
        except:
            part_obj = {"__ERROR__": "parse failed"}

        # Compute TED
        ted_dist, norm_dist, similarity = compute_ted_similarity(part_obj, gold_obj)

        # Store result
        results.append({
            'participantId': pid,
            'format': fmt,
            'strict_check': strict_result,
            'loose_check': loose_result,
            'final_check': final_check,
            'ted': ted_dist,
            'norm_ted': norm_dist,
            'similarity': similarity
        })

    return pd.DataFrame(results)



import numpy as np
import pandas as pd

def add_overall_rows(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df
        .groupby(['participantId', 'format'], as_index=False)
        .agg({
            'strict_check': 'mean',
            'final_check':  'mean',
            'ted':          'mean',
            'norm_ted':     'mean',
            'similarity':   'mean'
        })
    )

    agg['task_id']     = 'overall'
    agg['loose_check'] = np.nan   # 一律 NaN

    cols = df.columns.tolist()
    agg = agg[cols]

    return pd.concat([df, agg], ignore_index=True)


# debug_anova_errors(combined_df)

# participant = '''
# # 用户信息
# name = "John",      # 用户姓名
# age = 30

# # 技能列表
# skills = ["Python", "TOML", "DataViz"]  # 注释在数组末尾
# '''

# gold = '''
# name = "John"
# age = 30
# skills = ["Python", "TOML", "DataViz"]
# '''

# fmt = 'toml'


# strict_result = check_strict_syntax(participant, fmt)
# loose_result = check_loose_syntax(participant, fmt) if strict_result == 0 else 1
# print("Strict Check:", strict_result)
# print("Loose Check :", loose_result)

# gold_obj = parse_to_obj(gold, fmt)
# part_obj = parse_to_obj(participant, fmt) if loose_result == 1 else {}

# if loose_result == 1:
#     ted_dist, norm_dist, similarity = compute_ted_similarity(part_obj, gold_obj)
# else:
#     ted_dist, norm_dist, similarity = None, None, None

# print("Strict Check:", strict_result)
# print("Loose Check :", loose_result)
# print("TED         :", ted_dist)
# print("Norm Dist   :", norm_dist)
# print("Similarity  :", similarity)
