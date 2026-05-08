import re
from datetime import timedelta

def parse_time_to_timedelta(time_str: str) -> timedelta:
    """
    将非标准的时间字符串 (例如 '54h22min17s') 解析为 timedelta 对象。
    
    Args:
        time_str: 输入的时间字符串。
        
    Returns:
        一个代表该时长的 timedelta 对象。
    """
    hours = 0
    minutes = 0
    seconds = 0
    
    # 使用正则表达式查找小时 (h)
    h_match = re.search(r'(\d+)\s*h', time_str)
    if h_match:
        hours = int(h_match.group(1))
        
    # 使用正则表达式查找分钟 (min)
    m_match = re.search(r'(\d+)\s*min', time_str)
    if m_match:
        minutes = int(m_match.group(1))
        
    # 使用正则表达式查找秒 (s)
    s_match = re.search(r'(\d+)\s*s', time_str)
    if s_match:
        seconds = int(s_match.group(1))
        
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)

def format_timedelta_to_str(td: timedelta) -> str:
    """
    将 timedelta 对象格式化为 'Xh Ymin Zs' 格式的字符串。
    
    Args:
        td: 输入的 timedelta 对象。
        
    Returns:
        格式化后的时间字符串。
    """
    # 获取总秒数
    total_seconds = int(td.total_seconds())
    
    # 计算小时、分钟和秒
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # 构建结果字符串
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}min")
    if seconds > 0:
        parts.append(f"{seconds}s")
        
    return "".join(parts) if parts else "0s"

def add_two_str_time_to_str(time_str1: str, time_str2: str) -> str:
    """
    将两个非标准时间字符串相加，并返回结果字符串。
    
    Args:
        time_str1: 第一个时间字符串。
        time_str2: 第二个时间字符串。
        
    Returns:
        相加后的时间字符串。
    """
    td1 = parse_time_to_timedelta(time_str1)
    td2 = parse_time_to_timedelta(time_str2)
    
    result_td = td1 + td2
    return format_timedelta_to_str(result_td)

if __name__ == '__main__':
    # --- 示例 1 ---
    # start_time_1 = "54h22min"
    start_time_1 = "0h0min0s"
    add_time_1 = "77h18min17s"

    # 1. 解析为 timedelta 对象
    td1_start = parse_time_to_timedelta(start_time_1)
    td1_add = parse_time_to_timedelta(add_time_1)

    # 2. 直接相加
    result_td_1 = td1_start + td1_add

    # 3. 格式化输出
    formatted_result_1 = format_timedelta_to_str(result_td_1)

    print(f"--- 计算示例 1 ---")
    print(f"开始时间: {start_time_1} ({td1_start})")
    print(f"相加时间: {add_time_1} ({td1_add})")
    print(f"结果: {formatted_result_1} (总计: {result_td_1})")
    print(f"结果2: {add_two_str_time_to_str(start_time_1, add_time_1)}")


    print("\n" + "="*30 + "\n")


    # --- 示例 2 ---
    start_time_2 = "54h22min"
    add_time_2 = "30min6s"

    # 1. 解析
    td2_start = parse_time_to_timedelta(start_time_2)
    td2_add = parse_time_to_timedelta(add_time_2)

    # 2. 相加
    result_td_2 = td2_start + td2_add

    # 3. 格式化
    formatted_result_2 = format_timedelta_to_str(result_td_2)


    print(f"--- 计算示例 2 ---")
    print(f"开始时间: {start_time_2} ({td2_start})")
    print(f"相加时间: {add_time_2} ({td2_add})")
    print(f"结果: {formatted_result_2} (总计: {result_td_2})")
    print(f"结果2: {add_two_str_time_to_str(start_time_2, add_time_2)}")