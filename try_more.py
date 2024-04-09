s = 'hello world hello everyone'

# 要查找的子字符串
substring = 'hello'

# 初始化起始位置
start = 0

# 使用 rfind 方法查找所有匹配项的起始位置
while True:
    position = s.find(substring, start)
    if position == -1:
        break  # 没有找到更多的匹配项
    print(f"The substring '{substring}' is found at position {position}.")
    start = position + 1  # 移动到下一个位置继续查找

def findall(substring):
    positions = []
    start = 0

    # 使用 find 方法查找所有匹配项的起始位置
    while True:
        position = s.find(substring, start)
        if position == -1:
            break  # 没有找到更多的匹配项
        positions.append(position)
        start = position + 1  # 移动到下一个位置继续查找
    
    return positions