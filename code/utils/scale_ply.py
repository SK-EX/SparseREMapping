"""按比例（等比 / 均匀）缩放 PLY 文件的 x/y/z 坐标，其余属性保持不变。

特性：
  * 支持 ascii、binary_little_endian、binary_big_endian 三种格式；
  * 完整解析头部并原样写回（元素顺序、注释等均保留）；
  * 仅对元素中名为 x / y / z 的*标量*坐标乘以统一缩放因子；
  * 法线 nx/ny/nz 及颜色等其它属性**不做**缩放（等比缩放不改法线方向，
    若法线是单位向量，缩放后依然有效）；
  * 面片（list 属性）等非几何元素原样复制。

用法：
    python scale_ply.py  <输入.ply>  <输出.ply>  <缩放因子>

示例：
    python scale_ply.py Data.ply Data_scaled.ply 0.22
"""
import sys
import struct
import numpy as np

# ---- 类型映射 -------------------------------------------------------------
# 属性类型 -> struct 字符 / numpy dtype 字符
STRUCT = {"char": "b", "uchar": "B", "short": "h", "ushort": "H",
          "int": "i", "uint": "I", "float": "f", "double": "d"}
NP = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
      "int": "i4", "uint": "u4", "float": "f4", "double": "f8"}
ENDIAN = {"binary_little_endian": "<", "binary_big_endian": ">", "ascii": ""}
GEOM = {"x", "y", "z"}   # 参与几何缩放的属性名
FLOAT_TYPES = {"float", "double"}


class Prop:
    """PLY 属性：kind = 'scalar' 或 'list'。"""
    __slots__ = ("kind", "type", "name", "count_type")

    def __init__(self, kind, type_, name, count_type=None):
        self.kind = kind
        self.type = type_
        self.name = name
        self.count_type = count_type

    def __repr__(self):
        if self.kind == "scalar":
            return f"scalar {self.type} {self.name}"
        return f"list {self.count_type} {self.type} {self.name}"


def parse_header(raw: bytes):
    """解析 PLY 头部，返回 (格式名, 元素描述列表)。"""
    lines = raw.decode("latin-1").splitlines()
    fmt = None
    elements = []
    cur = None
    for ln in lines:
        if ln == "ply" or ln.startswith("comment") or ln.startswith("obj_info"):
            continue
        if ln.startswith("format"):
            fmt = ln.split()[1]
            continue
        if ln.startswith("element"):
            cur = {"name": ln.split()[1], "count": int(ln.split()[2]),
                   "props": [], "dtype": None, "has_list": False}
            elements.append(cur)
            continue
        if ln.startswith("property"):
            parts = ln.split()
            if parts[1] == "list":
                cur["has_list"] = True
                cur["props"].append(Prop("list", parts[3], parts[4], parts[2]))
            else:
                cur["props"].append(Prop("scalar", parts[1], parts[2]))
            continue
        # 'end_header' 及空行等
    return fmt, elements


def build_dtypes(elements, endian):
    """为所有“纯标量”元素构造结构化 numpy dtype（命名与声明一致）。"""
    for el in elements:
        if el["has_list"]:
            continue
        fields = []
        for p in el["props"]:
            if p.kind == "scalar":
                fields.append((p.name, endian + NP[p.type]))
        el["dtype"] = np.dtype(fields)


# ---- 读取 ----------------------------------------------------------------
def load(path):
    """读取 PLY，返回 (header_bytes, format, out_elements)。

    每个元素按两种内部表示之一存储：
      * arr   —— numpy 结构化数组（纯标量属性元素）；
      * rows  —— 行列表，每行与 props 对齐（含 list 属性元素）。
    """
    with open(path, "rb") as f:
        header = bytearray()
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY 头部不完整（缺少 end_header）")
            header += line
            if line.strip() == b"end_header":
                break
        fmt, elements = parse_header(bytes(header))
        if fmt not in ENDIAN:
            raise ValueError(f"不支持的 PLY 格式: {fmt}")
        endian = ENDIAN[fmt]
        build_dtypes(elements, endian)
        body = f.read()

    if fmt == "ascii":
        # 解码为文本再切分，保证 token 为 str（np.array(str 列表) 才能解析数值）
        _load_ascii(body.decode("latin-1").split(), elements)
    else:
        _load_binary(body, elements, endian)

    out = [{"name": e["name"], "props": e["props"]} for e in elements]
    for i, e in enumerate(elements):
        if e["has_list"]:
            out[i]["rows"] = e["rows"]
        else:
            out[i]["arr"] = e["arr"]
    return bytes(header), fmt, out


def _load_ascii(tokens, elements):
    """ASCII：正文按空白拆成 token 流，按元素顺序顺序消费。"""
    ti = 0
    for el in elements:
        count = el["count"]
        props = el["props"]
        if el["has_list"]:
            rows = []
            for _ in range(count):
                rec = []
                for p in props:
                    if p.kind == "list":
                        n = int(tokens[ti]); ti += 1
                        rec.append([_conv(tokens[ti + k], p.type)
                                    for k in range(n)])
                        ti += n
                    else:
                        rec.append(_conv(tokens[ti], p.type)); ti += 1
                rows.append(rec)
            el["rows"] = rows
        else:
            nprops = len(props)
            total = nprops * count
            chunk = tokens[ti:ti + total]
            arr = np.empty(count, dtype=el["dtype"])
            for j, p in enumerate(props):
                col = np.array(chunk[j::nprops], dtype=NP[p.type])
                arr[p.name] = col
            el["arr"] = arr
            ti += total
    if ti != len(tokens):
        raise ValueError("ASCII 数据 token 数与头部声明不一致"
                         f"（{ti} != {len(tokens)}）")


def _load_binary(body, elements, endian):
    """二进制：按字节游标读取；list 属性逐条解析。"""
    pos = 0
    for el in elements:
        count = el["count"]
        props = el["props"]
        if el["has_list"]:
            rows = []
            for _ in range(count):
                rec = []
                for p in props:
                    if p.kind == "list":
                        n = struct.unpack_from(endian + STRUCT[p.count_type],
                                                body, pos)[0]
                        pos += struct.calcsize(endian + STRUCT[p.count_type])
                        vals = list(struct.unpack_from(
                            endian + STRUCT[p.type] * n, body, pos))
                        pos += n * struct.calcsize(endian + STRUCT[p.type])
                        rec.append(vals)
                    else:
                        val = struct.unpack_from(endian + STRUCT[p.type],
                                                 body, pos)[0]
                        pos += struct.calcsize(endian + STRUCT[p.type])
                        rec.append(val)
                rows.append(rec)
            el["rows"] = rows
        else:
            nbytes = count * el["dtype"].itemsize
            arr = np.frombuffer(body, dtype=el["dtype"], count=count, offset=pos)
            el["arr"] = arr.copy()
            pos += nbytes
    if pos != len(body):
        raise ValueError(f"二进制数据字节数与头部声明不一致"
                         f"（{pos} != {len(body)}）")


def _conv(tok, ptype):
    return float(tok) if ptype in FLOAT_TYPES else int(tok)


# ---- 缩放 ----------------------------------------------------------------
def apply_scale(out_elements, factor):
    """把各元素中名为 x/y/z 的标量坐标统一乘以 factor。"""
    for el in out_elements:
        if "arr" in el:
            for p in el["props"]:
                if p.kind == "scalar" and p.name in GEOM:
                    el["arr"][p.name] *= factor
            el["range"] = {p.name: (float(el["arr"][p.name].min()),
                                    float(el["arr"][p.name].max()))
                           for p in el["props"]
                           if p.kind == "scalar" and p.name in GEOM}
        else:
            # rows 分支：仅缩放标量坐标（list 属性原样保留）
            idxs = [i for i, p in enumerate(el["props"])
                    if p.kind == "scalar" and p.name in GEOM]
            if idxs:
                for rec in el["rows"]:
                    for i in idxs:
                        rec[i] = rec[i] * factor


# ---- 写回 ----------------------------------------------------------------
def _fmt_num(ptype, v):
    """把数值转成简洁的 ASCII 表示（整数型属性直接输出整数）。"""
    if ptype in FLOAT_TYPES:
        f = float(v)
        if f == int(f) and abs(f) < 1e15:
            return str(int(f))
        return str(np.float32(f)) if ptype == "float" else str(np.float64(f))
    return str(int(v))


def save(path, header, fmt, out_elements):
    with open(path, "wb") as f:
        f.write(header)
        if fmt == "ascii":
            _write_ascii(f, out_elements)
        else:
            endian = ENDIAN[fmt]
            _write_binary(f, out_elements, endian)


def _write_ascii(f, out_elements):
    for el in out_elements:
        if "arr" in el:
            arr, props = el["arr"], el["props"]
            for i in range(len(arr)):
                line = b" ".join(_fmt_num(p.type, arr[p.name][i]).encode()
                                 for p in props)
                f.write(line + b"\n")
        else:
            for rec in el["rows"]:
                toks = []
                for p, val in zip(el["props"], rec):
                    if p.kind == "list":
                        toks.append(str(len(val)))
                        toks.extend(_fmt_num(p.type, x) for x in val)
                    else:
                        toks.append(_fmt_num(p.type, val))
                f.write(b" ".join(t.encode() for t in toks) + b"\n")


def _write_binary(f, out_elements, endian):
    for el in out_elements:
        if "arr" in el:
            # 结构化数组的字段内存布局与声明顺序一致，直接写回即可
            f.write(el["arr"].tobytes())
        else:
            for rec in el["rows"]:
                for p, val in zip(el["props"], rec):
                    if p.kind == "list":
                        f.write(struct.pack(endian + STRUCT[p.count_type],
                                            len(val)))
                        for x in val:
                            f.write(struct.pack(endian + STRUCT[p.type], x))
                    else:
                        f.write(struct.pack(endian + STRUCT[p.type], val))


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    src, dst, factor = sys.argv[1], sys.argv[2], float(sys.argv[3])
    if factor <= 0:
        raise ValueError("缩放因子应为正数")

    header, fmt, elements = load(src)
    apply_scale(elements, factor)
    save(dst, header, fmt, elements)

    print(f"PLY 格式 : {fmt}")
    print(f"缩放因子 : {factor}")
    for el in elements:
        n = len(el["arr"]) if "arr" in el else len(el["rows"])
        info = f"  element {el['name']}  x{n}"
        if "range" in el:
            r = el["range"]
            x0, x1 = r["x"]; y0, y1 = r["y"]; z0, z1 = r["z"]
            info += (f"\n      X: [{x0:.4f}, {x1:.4f}]   "
                     f"Y: [{y0:.4f}, {y1:.4f}]   Z: [{z0:.4f}, {z1:.4f}]")
        print(info)
    print(f"已保存 : {dst}")


if __name__ == "__main__":
    main()
