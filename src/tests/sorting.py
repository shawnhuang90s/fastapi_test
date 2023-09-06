# -*- coding: utf-8 -*-
# @Time: 2023/9/6 14:49
"""
常见的时间复杂度量级（不考虑系数）：
    常数阶O(1)
        没有循环
    对数阶O(logN)
        二分查找
    线性阶O(n)
        单层循环
    线性对数阶O(nlogN)
        快排
    平方阶O(n²)
        嵌套双层循环
    立方阶O(n³)
    K次方阶O(n^k)
    指数阶(2^n)

时间复杂度对应算法的运行速度
空间复杂度对应算法占用的内存空间
"""

l = [3, 5, 1, 4, 7, 6, 9, 10]


def insertion_sort(l):
    """
    插入排序
        从第二个元素开始和前面的元素进行比较
        如果第一个元素比第二个元素大，则第二个元素与第一个元素位置互换
        依次类推，直到最后一个元素
    时间复杂度：平方阶O(n²)
    """
    for i in range(1, len(l)):
        current_num = l[i]
        pre_index = i - 1
        while pre_index >= 0 and l[pre_index] > current_num:
            l[pre_index + 1] = l[pre_index]
            pre_index -= 1
        l[pre_index + 1] = current_num
    return l


r = insertion_sort(l)
print(f"插入排序：{r}")


def selection_sort(l):
    """
    选择排序
        设第一个元素下标为最小元素下标 min_index，依次和后面的元素比较
        如果当前元素比第一个元素更小，则将当前元素下标作为最小元素下标
        依次类推，比较完所有元素找到最小的元素，将它和第一个元素互换
        重复上述操作，我们找出第二小的元素和第二个位置的元素互换，以此类推
    时间复杂度：平方阶O(n²)
    """
    for i in range(len(l) - 1):
        min_index = i
        for j in range(i + 1, len(l)):
            if l[j] < l[min_index]:
                min_index = j
        l[min_index], l[i] = l[i], l[min_index]

    return l


r = selection_sort(l)
print(f"选择排序：{r}")


def bubble_sort(l):
    """
    冒泡排序
        从第一个和第二个开始比较，如果第一个比第二个大，则交换位置
        然后比较第二个和第三个，依次类推，经过第一轮后最大的元素已经排在最后
        重复上述操作，第二大的则会排在倒数第二的位置，依次类推
        只需重复 n-1 次即可完成排序，因为最后一次只有一个元素
    时间复杂度：平方阶O(n²)
    """
    for i in range(len(l) - 1):
        for j in range(len(l) - 1 - i):
            if l[j] > l[j + 1]:
                l[j], l[j + 1] = l[j + 1], l[j]
    return l


r = bubble_sort(l)
print(f"冒泡排序：{r}")


def quick_sort(l):
    """
    快速排序
    时间复杂度：线性对数阶O(nlogN)
    """
    if len(l) < 2:
        return l
    else:
        pivot = l[0]
        less = [i for i in l[1:] if i <= pivot]
        greater = [i for i in l[1:] if i > pivot]
    return quick_sort(less) + [pivot] + quick_sort(greater)


r = quick_sort(l)
print(f"快速排序：{r}")
