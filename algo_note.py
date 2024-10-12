
#BUBBLE SORT
def bubbleSort(arr):
  for i in range(len(arr)-1,0,-1):
    for j in range(i):
      if arr[j]>arr[j+1]:
        arr[j],arr[j+1] = arr[j+1],arr[j]
  return arr

#Better version of Bubble Sort

def bubbleSort(arr):
  for i in range(len(arr)-1,0,-1):
    count = 0
    for j in range(i):
      if arr[j]>arr[j+1]:
        count+=1
        arr[j],arr[j+1] = arr[j+1],arr[j]
    if count==0:
      return arr
  return arr

"""Here whenever the array will become sorted, the function will return the array. So if the array is already sorted to begin with, the function will return the array and the algorithm will run in O(n) tc instead of O(n^2)."""

#SELECTION SORT

def selection_sort(arr):
  for i in range(len(arr)-2):
    min_index = i
    for j in range(i+1,len(arr)):
      if arr[j]<arr[min_index]:
        min_index = j
    arr[i],arr[min_index] = arr[min_index],arr[i]
  return arr

#DEVIDE AND CONQUER

#BINARY SEARCH

#Recursive
def bs(arr,val):
  mid = len(arr)//2
  if arr[mid]==val:
    return True
  elif len(arr)==1:
    return False
  elif arr[mid]>val:
    return bs(arr[:mid],val)
  elif arr[mid]<val:
    return bs(arr[mid:],val)

print(bs([2,2,3,3,5,8,9,12,14,17],9))

#Iterative
def bin(arr,val):
  first = 0
  last = len(arr)-1
  mid = (last+first)//2
  while first<=last:
    mid = (last+first)//2
    if arr[mid]==val:
      return mid
    elif arr[mid]<val:
      first = mid
    else:
      last = mid
  if arr[first]==val:
    return first
  elif arr[last]==val:
    return last
  return -1

print(bin([2,2,3,3,5,8,9,12,14,17],9))

"""If there is consecutive same value it will return the index of first appeared targeted value:"""

def bin(arr,val):
  first = 0
  last = len(arr)-1
  mid = (last+first)//2
  while first<=last:
    mid = (last+first)//2
    if arr[mid]==val:
      if mid-1 == 0:
        return mid
      elif arr[mid-1]<val:
        return mid
      else:
        last = mid
    elif arr[mid]<val:
      first = mid
    else:
      last = mid
  if arr[first]==val:
    return first
  elif arr[last]==val:
    return last
  return -1

print(bin([2,2,3,3,5,8,9,9,9,9,9,12,14,17],9))

#Ternary Search
def ternary_search(arr, val):
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3

        if arr[mid1] == val:
            return mid1
        elif arr[mid2] == val:
            return mid2

        if arr[mid1] < val:
            left = mid1 + 1
        elif arr[mid2] > val:
            right = mid2 - 1
        else:
            left = mid1 + 1
            right = mid2 - 1

    return -1

# Recurrence Relation:

# T(n) = T(n/3) + T(n/3) + c

# where:

# n is the size of the array.
# c is a constant representing the time taken to compare the value at the midpoints.

# The recurrence relation indicates that the running time of the ternary search algorithm is O(log3 n), where n is the size of the array. This is because the algorithm divides the search space into three parts at each step, reducing the size of the search space by a factor of 3.



#MERGE SORT"

def merge(left,right):
  new = []
  i,j = 0,0
  while i<len(left) and j<len(right):
    if left[i]<right[j]:
      new+=[left[i]]
      i+=1
    else:
      new+=[right[j]]
      j+=1
  if i==len(left):
    new+=right[j:]
  elif j==len(right):
    new+=left[i:]
  return new

def merge_sort(arr):
  if len(arr)<=1:
    return arr
  mid = len(arr)//2
  left = arr[:mid]
  right = arr[mid:]
  left_half = merge_sort(left)
  right_half = merge_sort(right)
  return merge(left_half, right_half)

print(merge_sort([2,50,1,56,4,0]))

#QUICK SORT

def quicksort(arr,s,e):
  if s<e:
    q = partition(arr,s,e)
    quicksort(arr,s,q-1)
    quicksort(arr,q+1,e)
  return arr

def partition(arr, low, high):
  pivot = arr[(low + high) // 2]
  i = low - 1
  j = high + 1
  while True:
    i += 1
    while arr[i] < pivot:
      i += 1
    j -= 1
    while arr[j] > pivot:
      j -= 1
    if i >= j:
      return j
    arr[i], arr[j] = arr[j], arr[i]


array = [2,2,50,1,56,4,0]
print(quicksort(array,0,len(array)-1))


#Maximum Subarray

#O(n^2)
arr=[5,4,-1,7,8]
m = arr[0]
for i in range(len(arr)-1):
  sum = 0
  for j in range(i,len(arr)):
    sum+=arr[j]
    if sum>m:
      m = sum
print(m)

#O(nlogn)
def max_subarray(arr):
  if len(arr) == 0:
    return None
  if len(arr) == 1:
    return arr[0]

  mid = len(arr) // 2
  left_max = max_subarray(arr[:mid])
  right_max = max_subarray(arr[mid:])
  cross_max = max_crossing_subarray(arr, mid)

  return max(left_max, right_max, cross_max)

def max_crossing_subarray(arr, mid):
  left_sum = float('-inf')
  sum = 0
  for i in range(mid - 1, -1, -1):
    sum += arr[i]
    left_sum = max(left_sum, sum)

  right_sum = float('-inf')
  sum = 0
  for i in range(mid, len(arr)):
    sum += arr[i]
    right_sum = max(right_sum, sum)

  return left_sum + right_sum

#O(n)
def maxSubArray(nums):
  max_sum = nums[0]
  cur_sum = 0
  cur_idx = []

  for n in range(len(nums)):
    if cur_sum<0:
      cur_sum = 0
      cur_idx = []
    cur_sum += nums[n]
    cur_idx += [n]
    if cur_sum>max_sum:
      max_sum = cur_sum
      i = cur_idx[0]
      j = cur_idx[len(cur_idx)-1]

  max_arr = nums[i:j+1]

  print(f"sum: {max_sum}")
  return max_arr

print(maxSubArray([-2,1,-3,4,-1,2,1,-1,-1]))
print(maxSubArray([-2,-2,-1]))
print(maxSubArray([-2,10,1]))

#KARATSUBA

def karatsuba(x,y):
  if x<10 or y<10:
    return x*y
  else:
    if x>y:
      n = x
    else:
      n = y
    h = n//2
    a = x//(10**h)
    b = x%(10**h)
    c = y//(10**h)
    d = y%(10**h)
    ac = karatsuba(a,c)
    bd = karatsuba(b,d)
    ad_bc = karatsuba(a+b, c+d)-ac-bd
  return ac*(10**(2*h)) + ad_bc*(10**h)+bd

##GRAPH

#BFS

def bfs(graph,color,source):
  q = [1]
  color[source] = 1
  out = []

  while len(q)>0:
    val = q.pop(0)
    out+=[val]
    for i in graph[val]:
      if color[i]==0:
        q.append(i)
        color[i] = 1

  return out

#DFS

def dfs(graph, start_node):
  visited = set()
  order = []

  def dfs_recursive(node):
    visited.add(node)
    order.append(node)

    for neighbor in graph.get(node, []):
      if neighbor not in visited:
        dfs_recursive(neighbor)

  dfs_recursive(start_node)
  return order

#DIJKSTRA

import heapq as hq

def djk(src, graph):
  dist = [float("inf")]*(len(graph))
  dist[src] = 0
  q = []
  hq.heappush(q, (0,src))

  while q:
    current_weight, node = hq.heappop(q)
    for ver, weight in graph[node]:
      if (weight + current_weight) < dist[ver]:
        dist[ver] = weight + current_weight
        hq.heappush(q, (dist[ver],ver))

  return dist