import struct
import time
import re
import os
import sys
import numpy as np
import hashlib
# 新增：引入 scipy 的 CubicSpline
from scipy.interpolate import CubicSpline

# ==============================================================================
# 1. 基础工具函数
# ==============================================================================

def input_string(prompt, default_val):
    val = input(f"{prompt} (默认: {default_val}): ").strip()
    return val if val else default_val

def input_file_path(prompt, default_val=None):
    p_str = f"{prompt} (默认: {default_val}): " if default_val else f"{prompt}: "
    val = input(p_str).strip()
    if val:
        val = val.replace('"', '').replace("'", "")
    return val if val else default_val

def input_float(prompt, default_val=None):
    while True:
        p_str = f"{prompt} (默认: {default_val}): " if default_val is not None else f"{prompt}: "
        val_str = input(p_str).strip()
        if not val_str and default_val is not None:
            return default_val
        try:
            return float(val_str)
        except ValueError:
            print("❌ 输入无效，请输入数字。")

def float_to_s15fixed16(value):
    value = max(-32768.0, min(32767.9999, value))
    fixed_val = int(round(value * 65536.0))
    return struct.pack('>i', fixed_val)

def s15fixed16_to_float(bytes_data):
    val_int = struct.unpack('>i', bytes_data)[0]
    return val_int / 65536.0

def float_to_u8fixed8(value):
    fixed_val = int(round(value * 256.0))
    return struct.pack('>H', fixed_val)

# ==============================================================================
# 2. 核心数学算法：绝对 XYZ 矩阵法 (Absolute XYZ Matrix)
# ==============================================================================

def xyY_to_XYZ(x, y, Y):
    """将 xyY 转换为 XYZ 三刺激值"""
    if y == 0: return np.array([0.0, 0.0, 0.0])
    X = (x * Y) / y
    Z = ((1 - x - y) * Y) / y
    return np.array([X, Y, Z])

def calculate_matrix_absolute(native_data, target_data):
    """
    使用绝对 XYZ 法计算矩阵。
    完全信任 R, G, B 的 xyY 实测数据。
    M = [XYZ_r, XYZ_g, XYZ_b]
    """
    print("\n🧮 正在使用绝对 XYZ 矩阵法计算 (Trust RGB Y)...")
    
    # 1. 构建 Native 矩阵 (实测值)
    # 直接使用列向量 [Native_R_XYZ, Native_G_XYZ, Native_B_XYZ]
    m_native = np.array([
        xyY_to_XYZ(*native_data['R']),
        xyY_to_XYZ(*native_data['G']),
        xyY_to_XYZ(*native_data['B'])
    ]).T

    # 2. 构建 Target 矩阵 (目标值)
    # 直接使用列向量 [Target_R_XYZ, Target_G_XYZ, Target_B_XYZ]
    m_target = np.array([
        xyY_to_XYZ(*target_data['R']),
        xyY_to_XYZ(*target_data['G']),
        xyY_to_XYZ(*target_data['B'])
    ]).T

    print("   -> Native Matrix (Raw XYZ Columns):")
    print(m_native)
    print("   -> Target Matrix (Raw XYZ Columns):")
    print(m_target)

    # 3. 计算变换矩阵
    try:
        m_native_inv = np.linalg.inv(m_native)
        m_final = np.dot(m_native_inv, m_target)
        
        # 调试：检查白点映射情况
        # 理论上，Target 的 (1,1,1) 输入经过矩阵后，应该变成 Native 的 (R+G+B)
        target_sum = m_target.sum(axis=1) # R+G+B in Target space
        native_sum = m_native.sum(axis=1) # R+G+B in Native space
        print(f"   [Debug] Target RGB Sum (XYZ): {target_sum}")
        print(f"   [Debug] Native RGB Sum (XYZ): {native_sum}")
        
        return m_final
    except np.linalg.LinAlgError:
        print("❌ 错误: Native 矩阵奇异，无法求逆。请检查 RGB 坐标是否重合。")
        return np.eye(3)

# ==============================================================================
# 3. 读取逻辑 (TXT & ICC)
# ==============================================================================

def load_lut_from_txt(file_path, target_size=4096):
    print(f"\n📂 [Mode 1] 读取 LUT TXT: {file_path}")
    if not os.path.exists(file_path):
        print("❌ 文件不存在")
        return None

    raw_r, raw_g, raw_b = [], [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                nums = re.findall(r'\d+', line)
                if len(nums) >= 4:
                    # 假设格式是 Index R G B (16bit)
                    raw_r.append(float(nums[-3])/65535.0)
                    raw_g.append(float(nums[-2])/65535.0)
                    raw_b.append(float(nums[-1])/65535.0)
    except: return None

    if len(raw_r) < 2: return None
    
    print(f"⚡ 执行插值 (Cubic Spline + Clip): {len(raw_r)} -> {target_size}")
    
    # 原始坐标
    x_src = np.linspace(0, 1, len(raw_r))
    # 目标坐标
    x_dst = np.linspace(0, 1, target_size)

    # --- 修改部分开始: 使用 CubicSpline 替代 np.interp ---
    
    # bc_type='natural' 确保两端二阶导数为0，防止端点剧烈震荡
    cs_r = CubicSpline(x_src, raw_r, bc_type='natural')
    cs_g = CubicSpline(x_src, raw_g, bc_type='natural')
    cs_b = CubicSpline(x_src, raw_b, bc_type='natural')

    # 计算插值
    new_r = cs_r(x_dst)
    new_g = cs_g(x_dst)
    new_b = cs_b(x_dst)

    # 强制钳位 (Clip) 到 [0, 1]，防止三次样条产生的过冲/下冲导致 ICC 数据损坏
    new_r = np.clip(new_r, 0.0, 1.0)
    new_g = np.clip(new_g, 0.0, 1.0)
    new_b = np.clip(new_b, 0.0, 1.0)
    
    # --- 修改部分结束 ---

    return (new_r, new_g, new_b)

def extract_luts_from_icc(icc_path):
    print(f"\n📂 [Mode 2] 解析 ICC 文件: {icc_path}")
    if not os.path.exists(icc_path): return None

    try:
        with open(icc_path, 'rb') as f: data = f.read()

        tag_count = struct.unpack('>I', data[128:132])[0]
        mhc2_offset = 0
        for i in range(tag_count):
            base = 132 + (i * 12)
            sig = data[base:base+4]
            if sig == b'MHC2':
                mhc2_offset = struct.unpack('>I', data[base+4:base+8])[0]
                break
        
        if mhc2_offset == 0: return None

        mhc_base = mhc2_offset
        lut_count = struct.unpack('>I', data[mhc_base+8:mhc_base+12])[0]
        min_lum = s15fixed16_to_float(data[mhc_base+12:mhc_base+16])
        peak_lum = s15fixed16_to_float(data[mhc_base+16:mhc_base+20])
        
        off_r = struct.unpack('>I', data[mhc_base+24:mhc_base+28])[0]
        off_g = struct.unpack('>I', data[mhc_base+28:mhc_base+32])[0]
        off_b = struct.unpack('>I', data[mhc_base+32:mhc_base+36])[0]

        def read_ch(rel_off, count):
            s = mhc_base + rel_off + 8
            vals = []
            for i in range(count):
                vals.append(s15fixed16_to_float(data[s+i*4:s+i*4+4]))
            return np.array(vals)

        return read_ch(off_r, lut_count), read_ch(off_g, lut_count), read_ch(off_b, lut_count), min_lum, peak_lum

    except Exception as e:
        print(f"❌ 解析 ICC 失败: {e}")
        return None

# ==============================================================================
# 4. ICC 生成 (Writer)
# ==============================================================================

def create_mluc_tag(text):
    b_text = text.encode('utf-16-be')
    d = bytearray(b'mluc\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x0c\x65\x6e\x55\x53')
    d.extend(struct.pack('>I', len(b_text))); d.extend(struct.pack('>I', 28)); d.extend(b_text)
    return d

def create_xyz_tag(xyz):
    d = bytearray(b'XYZ \x00\x00\x00\x00')
    for v in xyz: d.extend(float_to_s15fixed16(v))
    return d

def create_chad_tag():
    d = bytearray(b'sf32\x00\x00\x00\x00')
    for v in [1.0,0,0, 0,1.0,0, 0,0,1.0]: d.extend(float_to_s15fixed16(v))
    return d

def create_curve_tag_gamma(g=2.2):
    return bytearray(b'curv\x00\x00\x00\x00\x00\x00\x00\x01') + float_to_u8fixed8(g)

def create_mhc2_tag_data(matrix_3x3, lut_r, lut_g, lut_b, min_lum, peak_lum):
    matrix_data = bytearray()
    for row in matrix_3x3:
        for val in row: matrix_data.extend(float_to_s15fixed16(val))
        matrix_data.extend(b'\x00\x00\x00\x00')
    
    sf32_head = b'\x73\x66\x33\x32\x00\x00\x00\x00'
    luts_data = []
    for c in [lut_r, lut_g, lut_b]:
        chunk = bytearray(sf32_head)
        for v in c: chunk.extend(float_to_s15fixed16(v))
        luts_data.append(chunk)
        
    mhc2 = bytearray(b'MHC2'); mhc2.extend(struct.pack('>I', 0))
    mhc2.extend(struct.pack('>I', len(lut_r)))
    mhc2.extend(float_to_s15fixed16(min_lum)); mhc2.extend(float_to_s15fixed16(peak_lum))
    
    off_mat = 36
    off_r = off_mat + len(matrix_data)
    off_g = off_r + len(luts_data[0])
    off_b = off_g + len(luts_data[1])
    
    mhc2.extend(struct.pack('>I', off_mat)); mhc2.extend(struct.pack('>I', off_r))
    mhc2.extend(struct.pack('>I', off_g)); mhc2.extend(struct.pack('>I', off_b))
    
    mhc2.extend(matrix_data)
    mhc2.extend(luts_data[0]); mhc2.extend(luts_data[1]); mhc2.extend(luts_data[2])
    return mhc2

def save_icc_profile(config, matrix_3x3, lut_r, lut_g, lut_b):
    filename = config['filename']
    print(f"\n📦 生成 ICC: {filename}")

    author_str = f"Copyright {config['author']}"

    tags = [
        (b'desc', create_mluc_tag(config['desc'])),
        (b'cprt', create_mluc_tag(author_str)),
        (b'wtpt', create_xyz_tag([0.9642, 1.0000, 0.8249])),
        (b'rXYZ', create_xyz_tag([0.4361, 0.2225, 0.0139])), 
        (b'gXYZ', create_xyz_tag([0.3851, 0.7169, 0.0971])),
        (b'bXYZ', create_xyz_tag([0.1431, 0.0606, 0.7139])),
        (b'rTRC', create_curve_tag_gamma()), (b'gTRC', create_curve_tag_gamma()), (b'bTRC', create_curve_tag_gamma()),
        (b'chad', create_chad_tag()),
        (b'MHC2', create_mhc2_tag_data(matrix_3x3, lut_r, lut_g, lut_b, config['min'], config['peak']))
    ]
    tags.sort(key=lambda x: x[0])

    tag_count = len(tags)
    data_offset = 128 + 4 + (12 * tag_count)
    
    table = bytearray(); body = bytearray()
    curr_off = data_offset
    table.extend(struct.pack('>I', tag_count))
    
    for sig, d in tags:
        while curr_off % 4 != 0: body.extend(b'\x00'); curr_off += 1
        table.extend(sig); table.extend(struct.pack('>I', curr_off)); table.extend(struct.pack('>I', len(d)))
        body.extend(d); curr_off += len(d)

    full_content = table + body
    total_size = 128 + len(full_content)
    
    header = bytearray(128)
    struct.pack_into('>I', header, 0, total_size)
    struct.pack_into('>4s', header, 4, b'\x00\x00\x00\x00')
    struct.pack_into('>I', header, 8, 0x04200000)
    struct.pack_into('>4s', header, 12, b'mntr'); struct.pack_into('>4s', header, 16, b'RGB '); struct.pack_into('>4s', header, 20, b'XYZ ')
    t = time.localtime()
    for i,v in enumerate([t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec]): struct.pack_into('>H', header, 24+i*2, v)
    struct.pack_into('>4s', header, 36, b'acsp'); struct.pack_into('>4s', header, 40, b'MSFT')
    struct.pack_into('>i', header, 68, 63190); struct.pack_into('>i', header, 72, 65536); struct.pack_into('>i', header, 76, 54060)
    struct.pack_into('>4s', header, 80, b'GMNI')
    
    temp_h = bytearray(header); struct.pack_into('>16s', temp_h, 84, b'\x00'*16)
    md5 = hashlib.md5(temp_h + full_content).digest()
    struct.pack_into('>16s', header, 84, md5)

    with open(filename, 'wb') as f: f.write(header); f.write(full_content)
    print(f"🎉 成功! (Size: {total_size})")

# ==============================================================================
# 5. 主程序 (交互)
# ==============================================================================

def get_metadata_inputs(default_file_name):
    """手动元数据输入 (含作者)"""
    print("\n--- ⚙️ 元数据配置 (按回车使用默认值) ---")
    man = input_string("厂商 (Manufacturer)", "Gemini")
    model = input_string("型号 (Model)", "HDR Calibrated")
    author = input_string("作者 (Author)", "User")
    desc = input_string("ICC 说明 (Description)", f"{man} - {model}")
    min_lum = input_float("最低亮度 (Min Nits)", 0.005)
    peak_lum = input_float("峰值亮度 (Peak Nits)", 1000.0)
    fname = input_string("输出文件名", default_file_name)
    if not fname.endswith('.icc'): fname += ".icc"
    
    return {'filename': fname, 'man': man, 'model': model, 'author': author, 'desc': desc, 'min': min_lum, 'peak': peak_lum}

def get_wrgb_manual(title):
    print(f"\n--- 🎨 {title} 数据录入 (xyY) ---")
    print("📝 请输入完整的 xyY 数据。")
    print("   对于非线性设备(OLED/HDR)，请使用实测 R,G,B 亮度以获得最佳矩阵。")
    data = {}
    for c in ['R', 'G', 'B', 'W']:
        print(f"[{c} 通道]")
        x = input_float(f"  x")
        y = input_float(f"  y")
        Y = input_float(f"  Y (亮度nits)", default_val=100.0)
        data[c] = (x, y, Y)
    return data

def main():
    print("==================================================")
    print("   Windows HDR MHC2 ICC 生成器")
    print("==================================================")
    
    mode = input("\n请选择模式 [1:灰阶校准 / 2:矩阵校准]: ").strip()
    
    if mode == '1':
        # Step 1: Grayscale Setup
        txt = input_file_path("\n请拖入 LUT TXT 文件", "Default Unity.txt")
        res = load_lut_from_txt(txt)
        if not res: return
        
        cfg = get_metadata_inputs("Step1.icc")
        save_icc_profile(cfg, np.eye(3), res[0], res[1], res[2])
        
    elif mode == '2':
        # Step 2: Matrix Calibration
        icc = input_file_path("\n请拖入 Step 1 ICC 文件", "Step1.icc")
        res = extract_luts_from_icc(icc)
        if not res: return
        lut_r, lut_g, lut_b, min_l, peak_l = res
        
        print(f"\n检测到原 ICC 亮度范围: {min_l} - {peak_l} nits")
        cfg = get_metadata_inputs("Step2_Final.icc")

        # 全手动输入 (含RGB Y)
        native = get_wrgb_manual("实测 Native")
        target = get_wrgb_manual("目标 Target")
        
        # 计算 (绝对法)
        matrix = calculate_matrix_absolute(native, target)
        print("最终矩阵:\n", matrix)
        
        save_icc_profile(cfg, matrix, lut_r, lut_g, lut_b)
    
    else:
        print("无效输入")
        
    input("\n按回车退出...")

if __name__ == "__main__":
    main()
