def twoSum(nums, k):
    """Return True if two numbers in the array add up to k"""
  dictionary = {}

  for num in nums:
    if num in dictionary:
      return True
    else:
      dictionary[k-num] = True
  return False

print(twoSum([2,7,11,15], 9))

print("BREAK")

