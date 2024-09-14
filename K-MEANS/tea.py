


if __name__ == '__main__':
    num = []
    for i in range(10):
        si = input("输入数字")
        num.append(si)
    print(num)
    num.sort(reverse=True)
    print(num)