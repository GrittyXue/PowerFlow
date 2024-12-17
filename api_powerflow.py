import re
import numpy as np
import math
from scipy.sparse import csr_matrix


def get_pattern(matlab_code, pattern, k_values):
    """
    :param matlab_code: 读取到的原始.m文件内容
    :param pattern: 匹配的规则条件
    :param k_values:标幺值幅值信息输入，将矩阵中线路参数等效替换为标幺值下的参数
    :return:返回获取到的参数矩阵
    """
    # 检索获取原始数据，并转为python
    try:
        matches = re.search(pattern, matlab_code, re.DOTALL)
        if matches:
            # 按照空格分割
            bus_matrix = matches.group(1).strip()
            # 替换 k1 .... kn 的值
            if k_values is not None:
                k_count = len(k_values)
                bus_matrix = re.sub(r'\*k(\d+)',
                                    lambda match: f'*{k_values[int(match.group(1)) - 1]}' if 0 <= int(
                                        match.group(1)) - 1 < k_count else match.group(0),
                                    bus_matrix)
            # 处理每一行，分割数据并将其转换为 float
            bus_data = [[float(eval(value)) for value in line.split()] for line in bus_matrix.splitlines()]
            # 转为矩阵形式，返回
            bus_data_np = np.array(bus_data)
            return bus_data_np
        else:
            print("模式匹配失败")
            return None
    except Exception as e:
        print(e, "模式匹配失败")
        return None


def read_case(file_input):
    """
    读取网络结构及其参数，初始化矩阵
    :param file_input-输入数据为.m文件
    :return:
        bus_matrix--母线参数矩阵
        line_matrix--线路参数矩阵
        transform_matrix--变压器参数矩阵
    """
    # 打开并读取文件
    with open(file_input, 'r', encoding='utf-8') as file:
        matlab_code = file.read()
    try:
        # 标幺值信息读取
        ub_match = re.search(r'UB=(\d+\.?\d*);', matlab_code)
        sb_match = re.search(r'SB=(\d+\.?\d*);', matlab_code)
        ub_value = float(ub_match.group(1)) if ub_match else None
        sb_value = float(sb_match.group(1)) if sb_match else None
        print("获取到的标幺值", f"基准电压kV:{ub_value}", f"基准功率MVA:{sb_value}")
        # 提取kn的表达式
        k_matches = re.findall(r'k\d+=([^;]+);', matlab_code)
        # 将表达式中的UB^2替换为UB**2
        k_matches = [expr.replace('^', '**') for expr in k_matches]
        # 计算代入后的结果
        if k_matches:
            k_values = []
            for k_expr in k_matches:
                k_value = eval(k_expr, {"UB": ub_value, "SB": sb_value})
                k_values.append(k_value)
        else:
            k_values = None
    except Exception as error:  # 捕获所有异常，因为这里需要确保后续流程可以继续
        k_values = None
        print(f"获取k值失败，报错如下{error}")
    # pattern 匹配根据文件中的矩阵类型修改，这边只有Bus、Line、Transformer
    pattern = r"[Bb]us=\[(.*?)\];"
    pattern1 = r"[Ll]ine=\[(.*?)\];"
    pattern2 = r"[Tt]ransform=\[(.*?)\];"
    # 根据匹配模式，获取对应数据并分别生成 母线矩阵、线路参数矩阵，变压器参数矩阵
    bus_matrix = get_pattern(matlab_code, pattern, k_values)
    line_matrix = get_pattern(matlab_code, pattern1, k_values)
    transform_matrix = get_pattern(matlab_code, pattern2, k_values)
    return bus_matrix, line_matrix, transform_matrix


def generate_node_admittance(num_buses, line_m_input, transformer_m_input=None):
    """
    生成节点导纳矩阵
    :param num_buses: 母线节点数量
    :param line_m_input: 输入线路参数
    :param transformer_m_input: 输入变压器参数
    :return: 导纳矩阵Y
    """
    # 初始化 Y 矩阵
    y_matrix = np.zeros((num_buses, num_buses), dtype=np.complex128)

    # 填充 Y 矩阵
    for line in line_m_input:
        # 首节点  末节点  电阻/欧姆   电抗/欧姆     对地电纳b/2       对地电纳b/2
        start = line[0]
        end = line[1]
        resistance = line[2]
        reactance = line[3]
        shunt_b1 = line[4]
        shunt_b2 = line[5]
        # 首先计算阻抗，然后计算导纳
        impedance = complex(resistance, reactance)
        admittance = 1 / impedance
        # 调整为从 0 开始的索引
        start_index = int(start - 1)
        end_index = int(end - 1)
        # 更新 Y 矩阵
        # 考虑线路的充电导纳
        y_matrix[start_index, start_index] += complex(0, shunt_b1)
        y_matrix[end_index, end_index] += complex(0, shunt_b2)
        # 互导纳
        y_matrix[start_index, end_index] -= admittance
        y_matrix[end_index, start_index] -= admittance
        # 自导纳
        y_matrix[start_index, start_index] += admittance
        y_matrix[end_index, end_index] += admittance
    # 有变压器
    if transformer_m_input is not None:
        # 为每个变压器应用电力工程的计算
        for trans_deal in transformer_m_input:
            # 获取变压器的起始节点，终止节点和相关参数
            start_trans = trans_deal[0]  # 变压器起始节点编号
            end_trans = trans_deal[1]  # 变压器终止节点编号
            resistance_trans = trans_deal[2]  # 变压器电阻值
            reactance_trans = trans_deal[3]  # 变压器电抗值
            shunt_b1_trans = trans_deal[4]  # 变压器一侧的并联电纳
            shunt_b2_trans = trans_deal[5]  # 变压器另一侧的并联电纳
            k = trans_deal[6]  # 变压器变比（TAP）

            # 计算阻抗和导纳
            impedance_trans = complex(resistance_trans, reactance_trans)  # 构成复数阻抗
            admittance_trans = 1 / impedance_trans  # 计算导纳，为阻抗的倒数

            # 调整节点索引以适应以0为起点的数组索引
            start_index_trans = int(start_trans - 1)  # 将节点编号转换为数组索引
            end_index_trans = int(end_trans - 1)

            # 更新自导纳
            # 对应节点的自导纳增加，考虑变压器变比的影响
            y_matrix[start_index_trans, start_index_trans] += admittance_trans / (k * k)
            y_matrix[end_index_trans, end_index_trans] += admittance_trans

            # 更新互导纳
            # 对应的节点之间的互导纳减少，反映变压器的耦合作用
            y_matrix[start_index_trans, end_index_trans] -= admittance_trans / k
            y_matrix[end_index_trans, start_index_trans] -= admittance_trans / k
    return y_matrix


def pad_to_shape(matrix, target_shape):
    """
    将给定矩阵扩展至目标形状，必要时进行零填充
    :param matrix: numpy.ndarray待扩展的二维数组
    :param target_shape: tuple of int目标形状，以(行数, 列数)形式提供
    :return: numpy.ndarray扩展并填充到目标形状的矩阵
    """
    # 计算需要填充的行数
    rows_to_add = target_shape[0] - matrix.shape[0]
    if rows_to_add > 0:
        # 创建零填充
        padding = np.zeros((rows_to_add, matrix.shape[1]))
        # 在现有矩阵下方添加零行
        matrix = np.vstack([matrix, padding])

    # 检查并填充列（一般情况下行填充即可，除非存在列数不足）
    cols_to_add = target_shape[1] - matrix.shape[1]
    if cols_to_add > 0:
        padding = np.zeros((matrix.shape[0], cols_to_add))
        matrix = np.hstack([matrix, padding])
    return matrix


def get_jacobbi(jacobbi, p2ux, p2uy, q2ux, q2uy, u22ux, u22uy, index_pq, index_pv, pv_num, pq_num, num_buses):
    """
    生成雅各比矩阵，根据输入的雅各比矩阵刷新覆盖
    :param jacobbi:-输入的雅各比矩阵-处理前矩阵
    :param p2ux:-输入刷新后的雅各比矩阵组成-分割矩阵  有功功率对电压幅值的偏导数部分
    :param p2uy:-输入刷新后的雅各比矩阵组成-分割矩阵  有功功率对电压相角的偏导数部分
    :param q2ux:-输入刷新后的雅各比矩阵组成-分割矩阵  无功功率对电压幅值的偏导数部分
    :param q2uy:-输入刷新后的雅各比矩阵组成-分割矩阵  无功功率对电压相角的偏导数部分
    :param u22ux:-输入刷新后的雅各比矩阵组成-分割矩阵 表示对于PV节点，自身电压幅值对相应节点参数变化的影响
    :param u22uy:-输入刷新后的雅各比矩阵组成-分割矩阵    表示对于PV节点，电压相角对相应节点参数变化的影响
    :param index_pq:-PQ节点索引序列
    :param index_pv:-PV节点索引序列
    :param pv_num:-PV节点数量
    :param pq_num:-PQ节点数量
    :param num_buses:-总节点数据
    :return:jacobbi-返回刷新后的雅各比矩阵
    """
    """
    p2ux:
    解释：表示有功功率对电压幅值的偏导数部分（partial derivative of active power with respect to voltage magnitude）。
    用途：用于雅各比矩阵的第一块分段，描述了在有功功率平衡方程中，节点电压幅值发生变化时的变化率。
    p2uy:
    解释：表示有功功率对电压相角的偏导数部分（partial derivative of active power with respect to voltage angle）。
    用途：雅各比矩阵的第二块子矩阵，量化了有功功率平衡方程中，相角变化时的影响。
    q2ux:
    解释：表示无功功率对电压幅值的偏导数部分（partial derivative of reactive power with respect to voltage magnitude）。
    用途：雅各比矩阵的第三块分段，描述了在无功功率平衡方程中，电压幅值变化的影响。
    q2uy:
    解释：表示无功功率对电压相角的偏导数部分（partial derivative of reactive power with respect to voltage angle）。
    用途：雅各比矩阵的第四块子矩阵，描述了无功功率平衡方程中，相角变化时的影响。
    u22ux:
    解释：表示对于 PV 节点，自身电压幅值对相应节点参数变化的影响。具体实现可能涉及对角块，因为 PV 节点电压幅值恒定。
    用途：这个矩阵的元素主要用于维持 PV 节点的电压控制情况。
    u22uy:
    解释：表示对于 PV 节点，电压相角对相应节点参数变化的影响。可能也用于对 PV 节点的电压角度进行调整或保持。
    用途：配合 u22ux，去描述和调节 PV 节点处的状态。
    """
    try:
        # 生成雅各比矩阵
        sub_matrix = np.array(p2ux[np.ix_(index_pq, index_pq)], dtype=np.complex128)
        sub_matrix1 = np.array(p2uy[np.ix_(index_pq, index_pq)], dtype=np.complex128)
        sub_matrix2 = np.array(p2ux[np.ix_(index_pq, index_pv)], dtype=np.complex128)
        sub_matrix3 = np.array(p2uy[np.ix_(index_pq, index_pv)], dtype=np.complex128)

        sub_matrix4 = np.array(q2ux[np.ix_(index_pq, index_pq)], dtype=np.complex128)
        sub_matrix5 = np.array(q2uy[np.ix_(index_pq, index_pq)], dtype=np.complex128)
        sub_matrix6 = np.array(q2ux[np.ix_(index_pq, index_pv)], dtype=np.complex128)
        sub_matrix7 = np.array(q2uy[np.ix_(index_pq, index_pv)], dtype=np.complex128)

        sub_matrix8 = np.array(p2ux[np.ix_(index_pv, index_pq)], dtype=np.complex128)
        sub_matrix9 = np.array(p2uy[np.ix_(index_pv, index_pq)], dtype=np.complex128)
        sub_matrix10 = np.array(p2ux[np.ix_(index_pv, index_pv)], dtype=np.complex128)
        sub_matrix11 = np.array(p2uy[np.ix_(index_pv, index_pv)], dtype=np.complex128)

        if index_pv.size > 0:
            number = index_pv
            # 提取对角元素
            diagonal_elements = u22ux[index_pv, index_pv]
            # 将二维数组扁平化为一维数组
            diagonal_elements_flat = diagonal_elements.A1  # 转换为一维数组

            # 使用 np.diag() 创建一个 2x2 的对角矩阵
            sub_matrix12 = np.diag(diagonal_elements_flat)
            diagonal_elements = u22uy[index_pv, index_pv]
            # 将二维数组扁平化为一维数组
            diagonal_elements_flat = diagonal_elements.A1  # 转换为一维数组
            sub_matrix13 = np.diag(diagonal_elements_flat)
        else:
            sub_matrix12 = np.array([], dtype=u22ux.dtype).reshape(0, 0)
            sub_matrix13 = np.array([], dtype=u22uy.dtype).reshape(0, 0)
        sub_matrix14 = pad_to_shape(np.zeros((pv_num, pq_num)), [pv_num, pq_num])
        sub_matrix15 = pad_to_shape(np.zeros((pv_num, pq_num)), [pv_num, pq_num])

        # 确定子矩阵的位置和大小，例如sub_matrix放在jacobbi的(0:len(index_pq), 0:len(index_pq))位置
        if sub_matrix.size > 0:
            jacobbi[0:len(index_pq), 0:len(index_pq)] = sub_matrix

        # sub_matrix1放在(jacobbi的(0:len(index_pq), len(index_pq):)
        if sub_matrix1.size > 0:
            jacobbi[0:len(index_pq), len(index_pq):(len(index_pq) * 2)] = sub_matrix1

        # sub_matrix2放在jacobbi的(len(index_pq):(len(index_pq) + len(index_pv)), 0:len(index_pq))位置
        if sub_matrix2.size > 0:
            number = (len(index_pq) * 2) + len(index_pv)
            jacobbi[0:len(index_pq), (len(index_pq) * 2):(len(index_pq) * 2) + len(index_pv)] = sub_matrix2

        # sub_matrix3放在(jacobbi的(len(index_pq):(len(index_pq) + len(index_pv)), len(index_pq):(len(index_pq) * 2))位置
        if sub_matrix3.size > 0:
            jacobbi[0:len(index_pq),
            (len(index_pq) * 2) + len(index_pv):(len(index_pq) * 2) + (len(index_pv) * 2)] = sub_matrix3

        # 继续插入新矩阵
        # sub_matrix4 放在雅各比矩阵的合适位置，注意调整行列下标
        if sub_matrix4.size > 0:
            jacobbi[len(index_pq):(2 * len(index_pq)), 0:len(index_pq)] = sub_matrix4

        if sub_matrix5.size > 0:
            jacobbi[len(index_pq):(2 * len(index_pq)), len(index_pq):(2 * len(index_pq))] = sub_matrix5

        if sub_matrix6.size > 0:
            jacobbi[len(index_pq):(2 * len(index_pq)),
            (len(index_pq) * 2):(len(index_pq) * 2) + len(index_pv)] = sub_matrix6

        if sub_matrix7.size > 0:
            jacobbi[len(index_pq):(2 * len(index_pq)),
            (len(index_pq) * 2) + len(index_pv):(len(index_pq) * 2) + (len(index_pv) * 2)] = sub_matrix7

        # # sub_matrix8 and onwards...
        # 这边是行向量
        print("sub_matrix8", sub_matrix8.size)
        print("sub", jacobbi[(2 * len(index_pq)):(2 * len(index_pq) + len(index_pv)), 0:len(index_pq)].size)

        if sub_matrix8.size > 0:
            jacobbi[(2 * len(index_pq)):(2 * len(index_pq) + len(index_pv)), 0:len(index_pq)] = sub_matrix8

        if sub_matrix9.size > 0:
            print("sub_matrix9", sub_matrix9)
            jacobbi[(2 * len(index_pq)):(2 * len(index_pq) + len(index_pv)),
            len(index_pq):(2 * len(index_pq))] = sub_matrix9
        # 空0*0
        if sub_matrix10.size > 0:
            jacobbi[(2 * len(index_pq)):(2 * len(index_pq) + len(index_pv)),
            (2 * len(index_pq)):(2 * len(index_pq) + len(index_pv))] = sub_matrix10

        if sub_matrix11.size > 0:
            print(sub_matrix11.size)
            jacobbi[(2 * len(index_pq)):(2 * len(index_pq) + len(index_pv)),
            (2 * len(index_pq) + len(index_pv)):(2 * (len(index_pq) + len(index_pv)))] = sub_matrix11

        # # Adjust remaining matrices as needed
        # # sub_matrix12 放到雅各比矩阵的(0, 0)起始或根据具体需要
        if sub_matrix12.size > 0:
            # 适当调整目标位置
            print(sub_matrix12.size)
            jacobbi[(2 * len(index_pq) + len(index_pv)):(2 * len(index_pq) + len(index_pv)) + pv_num,
            (2 * len(index_pq)):(2 * len(index_pq) + len(index_pv))] = sub_matrix12
            print("Fuck")

        if sub_matrix13.size > 0:
            print(sub_matrix13.size)
            # 适当调整目标位置
            jacobbi[(2 * len(index_pq) + len(index_pv)):(2 * len(index_pq) + len(index_pv)) + pv_num,
            (2 * len(index_pq) + len(index_pv)):(2 * (len(index_pq) + len(index_pv)))] = sub_matrix13

        # # sub_matrix14 和 sub_matrix15 需要根据其填充函数pad_to_shape调整位置和尺寸
        # 32*32
        if sub_matrix14.size > 0:
            print(sub_matrix14.size)
            jacobbi[(2 * len(index_pq) + len(index_pv)):(2 * len(index_pq) + len(index_pv)) + pv_num,
            (len(index_pq)):(2 * len(index_pq))] = sub_matrix14

        if sub_matrix15.size > 0:
            print(sub_matrix15.size)
            jacobbi[(2 * len(index_pq) + len(index_pv)):(2 * len(index_pq) + len(index_pv)) + pv_num,
            len(index_pq):2 * len(index_pq)] = sub_matrix15
        print("生成雅各比矩阵成功！")
    except Exception as e:
        print("生成雅可比矩阵失败！将返回原始雅各比矩阵")
        print("报错信息:", e)
    return jacobbi


def cal_power_flow_newton(y_matrix_input, bus_matrix_input, error_tol):
    """
    牛顿-拉夫逊法计算潮流
    :param y_matrix_input:-输入的导纳矩阵Y
    :param bus_matrix_input:-输入的节点矩阵Bus
    :param error_tol:-误差允许值
    :return:放回节点电压数据
    """
    # 获取节点数量
    num_bus = bus_matrix_input.shape[0]

    # 初始化迭代计数器
    circle_count = 0
    # 初始化迭代状态标志
    circle_status = True
    # 分离导纳矩阵的实部和虚部
    g_matrix = y_matrix_input.real  # 电导矩阵
    b_matrix = y_matrix_input.imag  # 电纳矩阵

    # 初始化雅各比矩阵，用于功率流计算
    jacobbi = np.zeros([(num_bus - 1) * 2, (num_bus - 1) * 2], dtype=np.complex128)

    # 电压的实部和虚部初始化
    voltage_e = np.zeros([num_bus], dtype=np.complex128)
    voltage_f = np.zeros([num_bus], dtype=np.complex128)

    # 功率修正量初始化
    power_correction = np.zeros([(num_bus - 1) * 2], dtype=np.complex128)

    # 注入的复功率初始化
    s = np.zeros([num_bus], dtype=np.complex128)

    # 单位电压及其平方的矢量初始化

    vv_correction = np.zeros([(num_bus - 1) * 2], dtype=np.complex128)  # voltage square correction term

    node_u_amp = np.zeros([num_bus], dtype=np.complex128)  # 电压赋值

    # 功率不平衡量初始化
    delta_pqu = np.zeros([(num_bus - 1) * 2], dtype=np.complex128)
    # 最终输出的节点电压
    u_final = np.zeros([num_bus], dtype=np.complex128)  # 或某个适合的默认值
    # 根据节点类型处理每个节点的电压和功率信息，并赋初值
    for node_num, bus in enumerate(bus_matrix_input):
        print(node_num)  # 打印当前节点编号
        node_type = int(bus[7])  # 获取节点类型
        # u = bus[1]  # 获取节点电压

        # 如果是平衡节点（Slack Bus）
        if node_type == 1:
            voltage_amp = bus[1].real  # 节点电压幅值
            voltage_angle = bus[2]  # 节点电压相角
            node_u_amp[node_num] = voltage_amp  # 记录保存节点电压幅值
            # 计算电压的复数形式
            voltage_comp = voltage_amp * complex(
                math.cos(voltage_angle / 180 * 3.14), math.sin(voltage_angle / 180 * 3.14)
            )
            voltage_e[node_num] = voltage_comp.real  # 电压实部
            voltage_f[node_num] = voltage_comp.imag  # 电压虚部

        # 如果是PV节点
        elif node_type == 3:
            voltage_amp = bus[1].real  # 电压幅值
            voltage_angle = bus[2]  # 电压相角
            # 有功功率修正量P=电源功率-负载功率
            power_correction[node_num] = (bus[3] - bus[5])
            # 注入的功率
            s[node_num] = (bus[3] - bus[5]) + complex(0, bus[4] - bus[6])
            node_u_amp[node_num] = voltage_amp  # 记录保存节点电压幅值
            voltage_comp = voltage_amp * complex(
                math.cos(voltage_angle / 180 * 3.14), math.sin(voltage_angle / 180 * 3.14)
            )
            voltage_e[node_num] = voltage_comp.real
            voltage_f[node_num] = voltage_comp.imag

        # 如果是PQ节点
        else:
            # 初始电压为1
            voltage_amp = bus[1].real
            # 有功功率修正量P=电源功率-负载功率
            power_correction[node_num] = (bus[3] - bus[5])
            # 注入的功率S
            s[node_num] = (bus[3] - bus[5]) + complex(0, bus[4] - bus[6])
            # 无功功率修正量Q=电源功率-负载功率
            power_correction[node_num + num_bus - 2] = (bus[4] - bus[6])
            node_u_amp[node_num] = voltage_amp  # 记录保存节点电压幅值
            voltage_e[node_num] = 1  # 初始电压实部
            voltage_f[node_num] = 0  # 初始电压虚部

    # 循环迭代
    while circle_status:
        # 提取电压实部和虚部并转为对角矩阵
        diag_ux = np.diag(voltage_e)
        diag_uy = np.diag(voltage_f)
        # 拼接实部和虚部
        complex_matrix = voltage_e + 1j * voltage_f
        # 转置矩阵，将原先行向量转为列向量，（1,num_bus-1）->（num_bus-1，1）
        ef_transpose = complex_matrix.reshape(-1, 1)
        # 计算电流 I=YU
        yu = np.dot(y_matrix_input, ef_transpose)
        n = num_bus
        # 雅各比矩阵组成部分刷新计算
        """
        p2ux:
        计算: p2ux = - np.dot(diag_ux, g_matrix) - np.dot(diag_uy, b_matrix) + csr_matrix((-yu.real.flatten(), (range(n), range(n))), shape=(n, n), dtype=np.complex128)
        解释: 计算有功功率对电压幅值变化的影响。该项涵盖了节点导纳矩阵的实部和虚部分量。
            diag_ux：电压幅值的对角阵。
            g_matrix：导纳矩阵的实部（电导部分）。
            b_matrix：导纳矩阵的虚部（电纳部分）。
            yu.real.flatten()：导纳矩阵实部的对角线元素。
        """
        p2ux = - np.dot(diag_ux, g_matrix) - np.dot(diag_uy, b_matrix) + csr_matrix(
            (-yu.real.flatten(), (range(n), range(n))),
            shape=(n, n), dtype=np.complex128)
        """
        p2uy:
        计算: p2uy = np.dot(diag_ux, b_matrix) - np.dot(diag_uy, g_matrix) + csr_matrix((-yu.imag.flatten(), (range(n), range(n))), shape=(n, n), dtype=np.complex128)
        解释: 计算有功功率对电压相角变化的影响。其中包括电纳矩阵对节点电压的旋转效应。
            yu.imag.flatten()：导纳矩阵虚部的对角线元素。
        """
        p2uy = np.dot(diag_ux, b_matrix) - np.dot(diag_uy, g_matrix) + csr_matrix(
            (-yu.imag.flatten(), (range(n), range(n))),
            shape=(n, n), dtype=np.complex128)
        """
        q2ux:
        计算: q2ux = np.dot(diag_ux, b_matrix) - np.dot(diag_uy, g_matrix) + csr_matrix((yu.imag.flatten(), (range(n), range(n))), shape=(n, n), dtype=np.complex128)
        解释: 计算无功功率对电压幅值变化的影响。主要考虑无功功率所受电压幅值的变化。
        该计算与 p2uy 类似，但带有不同符号用以表示 Q 的变化。
        """
        q2ux = np.dot(diag_ux, b_matrix) - np.dot(diag_uy, g_matrix) + csr_matrix(
            (yu.imag.flatten(), (range(n), range(n))),
            shape=(n, n), dtype=np.complex128)
        """
        q2uy:
        计算: q2uy = np.dot(diag_ux, g_matrix) + np.dot(diag_uy, b_matrix) + csr_matrix((-yu.real.flatten(), (range(n), range(n))), shape=(n, n), dtype=np.complex128)
        解释: 计算无功功率相对于电压相角的变化。此矩阵项调整与相角的相互作用。
        """
        q2uy = np.dot(diag_ux, g_matrix) + np.dot(diag_uy, b_matrix) + csr_matrix(
            (-yu.real.flatten(), (range(n), range(n))),
            shape=(n, n), dtype=np.complex128)
        """
        u22ux:
        计算: u22ux = csr_matrix((-2 * voltage_e, (range(n), range(n))), shape=(n, n), dtype=np.complex128)
        解释: PV 节点的电压幅值控件。其涉及电压的取值影响，是电压幅值的二次项对性能的直接影响。u22ux:
        计算: u22ux = csr_matrix((-2 * voltage_e, (range(n), range(n))), shape=(n, n), dtype=np.complex128)
        解释: PV 节点的电压幅值控件。其涉及电压的取值影响，是电压幅值的二次项对性能的直接影响。
        """
        u22ux = csr_matrix((-2 * voltage_e, (range(n), range(n))), shape=(n, n), dtype=np.complex128)
        """
        u22uy:
        计算: u22uy = csr_matrix((-2 * voltage_f, (range(n), range(n))), shape=(n, n), dtype=np.complex128)
        解释: PV 节点的电压相角控件。该项描述电压相角的变化幅度对节点复数电压的控制。
        """
        u22uy = csr_matrix((-2 * voltage_f, (range(n), range(n))), shape=(n, n), dtype=np.complex128)

        # 是否要考虑PV节点向PQ节点转化
        # 找到PQ节点的行索引
        pq_bus = np.where(bus_matrix_input[:, 7] == 2)[0]
        # 找到PV节点的行索引
        pv_bus = np.where(bus_matrix_input[:, 7] == 3)[0]
        pv_num = len(pv_bus)
        pq_num = len(pq_bus)
        index_pq = np.array(pq_bus).reshape(-1, 1).flatten()
        index_pv = np.array(pv_bus).reshape(-1, 1).flatten()
        # 计算偏差量-偏差量名称是否有问题
        vv_correction = node_u_amp * node_u_amp - (voltage_e * voltage_e + voltage_f * voltage_f)
        # 节点电压
        uu = voltage_e + 1j * voltage_f  # 节点电压
        # 功率修正量
        """
        power_correction 是节点期望功率与实际功率之间的差值。它在潮流计算的迭代过程中用于指导电压调整，以便最终达到系统的功率平衡。
        通过不断迭代，将 power_correction 降低到可接受的误差范围内，系统潮流计算即可视为收敛。
        """
        power_correction = s - uu * np.conj(np.dot(y_matrix_input, uu))
        # 使用电流方程计算得到实际功率与目标功率的差
        temp_matrix = power_correction[index_pq].real  # 提取PQ节点的有功部分
        temp_matrix1 = power_correction[index_pq].imag  # 提取PQ节点的无功部分
        temp_matrix2 = power_correction[index_pv].real  # 提取PV节点的有功部分
        temp_matrix3 = vv_correction[index_pv]  # PV节点的电压幅值修正量
        # 将上述各部分拼接成一个修正向量
        transposed_parts = np.concatenate([
            temp_matrix,  # 用于变压的修正值（有功部分）
            temp_matrix1,  # 用于变压的修正值（无功部分）
            temp_matrix2,  # 用于PV节点的修正值（有功部分）
            temp_matrix3  # 用于PV节点的电压修正
        ])
        # 将修正向量转换为列向量 delta_pqu，用于后续雅各比法求解
        delta_pqu = transposed_parts.reshape(-1, 1)

        # # 查找修正向量中绝对值最大的元素，用于判断收敛性
        overall_max = max(abs(delta_pqu))
        # 判断是否继续循环
        if abs(overall_max) > error_tol:
            if circle_count > 100:
                print("潮流不收敛！")
                # 停止迭代
                circle_status = False
            else:
                print(circle_count)
                circle_count += 1
                jacobbi = get_jacobbi(jacobbi, p2ux, p2uy, q2ux, q2uy, u22ux, u22uy, index_pq, index_pv, pv_num, pq_num,
                                      num_bus)
                # 牛拉法
                try:
                    circle_status = True
                    # 使用牛拉法（牛顿-拉夫森法）计算电压修正量 delta_u
                    delta_u = np.linalg.solve(jacobbi, -delta_pqu)
                    # 叠加修正量
                    # 将修正值应用到节点电压上 区间左闭右开
                    voltage_e[index_pq] += delta_u[0:pq_num].ravel()  # 更新PQ节点的有功电压部分
                    voltage_f[index_pq] += delta_u[pq_num:2 * pq_num].ravel()  # 更新PQ节点的无功电压部分
                    voltage_e[index_pv] += delta_u[2 * pq_num:2 * pq_num + pv_num].ravel()  # 更新PV节点的有功电压部分
                    voltage_f[index_pv] += delta_u[2 * pq_num + pv_num:2 * pq_num + 2 * pv_num].ravel()  # 更新PV节点的无功电压部分
                except Exception as erorr:
                    print(erorr)
                    print(circle_count)
                    circle_status = False
        else:
            # 如果所有功率修正量小于误差容限，认为迭代收敛，结束循环
            circle_status = False
            u_final = voltage_e + 1j * voltage_f
            print('循环结束！循环次数：', circle_count)
    return u_final


def calculate_power_loss(y_matrix, u_final_input, bus_matrix_input):
    """
    计算网络损耗
    :param bus_matrix_input:
    :param y_matrix: 输入导纳矩阵
    :param u_final_input: 输入潮流计算电压矩阵
    :return:返回功率损耗 power_loss
    """
    power_sum = np.conj(np.dot(y_matrix, u_final_input)) * u_final_input
    bus_type = bus_matrix_input[:, 7].astype(int)

    indices = np.where(bus_type == 1)[0]
    input_power = power_sum[0]
    mask = np.ones(bus_type.shape[0], dtype=bool)
    mask[indices] = False
    # 使用布尔索引选择要保留的行，然后求和
    sum_result = np.sum(power_sum[mask], dtype=np.complex128)
    power_loss = input_power + sum_result
    power_active = np.real(power_loss)
    power_reactive = np.imag(power_loss)
    return power_active, power_reactive


def calculate_voltage_bias(u_final_input, bus_matrix_input):
    """
    :param u_final_input:-输入潮流计算后的节点电压矩阵
    :param bus_matrix_input:-节点输入信息矩阵
    :return: voltage_bias - 电压偏差
    """
    voltage_bus = bus_matrix_input[:, 1]
    u_amp = np.abs(u_final_input)
    voltage_bias_list = u_amp - voltage_bus
    voltage_bias = np.sum(voltage_bias_list, dtype=np.float64)
    return voltage_bias


def power_flow(file_path_input, cp_matrix, pv_matrix):
    """
    潮流计算程序集合
    :param cp_matrix: -调节的电容器组容量-列向量输入
    :param pv_matrix: -调节的光伏参数容量-列向量输入
    :param file_path_input: 输入的参数文件
    :return:
        power_active-有功功率损耗MW
        voltage_bias-电压偏差（小数）
    """
    bus_demo, line_data, trans = read_case(file_path_input)
    num_columns = bus_demo.shape[0]  # 节点数量
    # 光伏列表
    pv_list = bus_demo[:, 8].astype(int)
    column_index = 4  # 无功功率
    # 光伏信息处理
    indices = np.where(pv_list == 1)[0]
    # 检查匹配条件
    if len(indices) == len(pv_matrix):
        # 如果匹配，则进行赋值操作
        bus_demo[indices, column_index] = pv_matrix
    else:
        # 如果不匹配，打印错误信息
        print("Error：光伏调节列向量的大小与行索引的大小不匹配！")
    # 电容器列表
    cp_list = bus_demo[:, 10].astype(int)
    # 光伏信息处理
    indices = np.where(cp_list == 1)[0]
    # 检查匹配条件
    if len(indices) == len(cp_matrix):
        # 如果匹配，则进行赋值操作
        bus_demo[indices, column_index] = cp_matrix
    else:
        # 如果不匹配，打印错误信息
        print("Error：电容器调节列向量的大小与行索引的大小不匹配！")
    # 生成节点导纳矩阵
    y_matrix1 = generate_node_admittance(num_columns, line_data, trans)
    # 潮流计算
    u_final = cal_power_flow_newton(y_matrix1, bus_demo, 0.000001)
    power_active, _ = calculate_power_loss(y_matrix1, u_final, bus_demo)
    voltage_bias = calculate_voltage_bias(u_final, bus_demo)
    return power_active, voltage_bias


if __name__ == '__main__':
    file_path = "./case33.m"
    power_flow(file_path, np.array([0, 0]), np.array([0, 0]))
