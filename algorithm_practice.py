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

def longestPalindromicSubstring(s):

    p = ''
    for i in range(len(s)):
        pOdd = get_palindrome(s, i, i)
        pEven = get_palindrome(s, i, i+1)
        print([p, pOdd, pEven])
        p = max([p, pOdd, pEven], key=lambda x: len(x))
        print('p is ', p)   
    return p

def get_palindrome(s, x, y):
    while x >= 0 and y < len(s) and s[x] == s[y]:
        x -= 1
        y += 1
        print('still in while loop', x, y)
    print('now out of loop', s[x+1:y])
    return s[x+1:y]

# print(longestPalindromicSubstring("babad")) # bab or aba
# print(longestPalindromicSubstring("cbbd")) # bb
# print(longestPalindromicSubstring("cyn")) # c
print(longestPalindromicSubstring("cccbaktkabd")) # baktkab

print("BREAK")

def lengthOfLongestSubstring(s):

    windowCharsMap = {}
    windowStart = 0
    maxLength = 0

    i = 0
    while i < len(s):
        endChar = s[i]

        if endChar in windowCharsMap and windowCharsMap[endChar] >= windowStart:
            windowStart = windowCharsMap[endChar] + 1

        windowCharsMap[endChar] = i
        maxLength = max(maxLength, i - windowStart + 1)

        i += 1

    return maxLength

print(lengthOfLongestSubstring('abcabcbb')) #3
print(lengthOfLongestSubstring('bbbbb'))    #1
print(lengthOfLongestSubstring('pwwkew'))   #3 wke
print(lengthOfLongestSubstring('bbbbyuhb'))  #4
print(lengthOfLongestSubstring('abcabcabcazbb')) #4
print(lengthOfLongestSubstring('abcheukls')) #9
print(lengthOfLongestSubstring('aabaab!bb')) #3
print(lengthOfLongestSubstring('abba')) #2

print("BREAK")

def houseRobber(nums):

    if len(nums) == 0:
        return 0
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums[0], nums[1])

    maxLootAtNth = [nums[0], max(nums[0], nums[1])]

    i = 2
    while i < len(nums):
        maxLootAtNth.append(max(nums[i] + maxLootAtNth[i-2], maxLootAtNth[i-1]))
        i += 1
    
    return maxLootAtNth.pop()

print(houseRobber([1,2,3,1])) #4
print(houseRobber([2,7,9,3,1])) #12
print(houseRobber([1,5,9,100,4,22,3])) #127

print("BREAK")

def canJump(nums):

    dpPositions = []
    for num in nums:
        dpPositions.append(False)
  
    dpPositions[0] = True

    j = 1
    while j < len(nums): 
        i = 0
        while i < j: 
            if dpPositions[i] and i + nums[i] >= j:
                dpPositions[j] = True
                # break
                i += 1
            i += 1
        j += 1
    return dpPositions[len(dpPositions) - 1]

print(canJump([2,3,1,1,4])) # True
print(canJump([3,2,1,0,4])) # False

print("BREAK")

# This version should be better
def canJump2(nums):
    ln = len(nums)
    can_reach = 0
    for idx, num in enumerate(nums):
        # i can't reach idx, then I can't move forward
        if idx > can_reach:
            return False

        can_reach = max(can_reach, idx + num) 
        #I just passed my destiny
        if can_reach >= ln - 1:
            return True

    return False

print(canJump2([2,3,1,1,4])) # True
print(canJump2([3,2,1,0,4])) # False

print("BREAK")

def firstMissingPositive(nums):

    if nums == []:
        return 1
    
    for num in range(1, max(nums) + 2):
        if num not in nums:
            return num

print(firstMissingPositive([])) #1
print(firstMissingPositive([1,2,0])) #3
print(firstMissingPositive([3,4,-1,1])) #2
print(firstMissingPositive([7,8,9,11,12])) #1

print("BREAK")

def firstMissingPositive2(nums):

    if nums == []:
        return 1
    
    newNums = set(nums)
    
    for num in range(1, max(nums) + 2):
        if num not in newNums:
            return num

print(firstMissingPositive2([])) #1
print(firstMissingPositive2([1,2,0])) #3
print(firstMissingPositive2([3,4,-1,1])) #2
print(firstMissingPositive2([7,8,9,11,12])) #1

print("BREAK")

def productExceptSelf(nums):

    output = []

    for x in nums:
        output.append(1)

    product = 1

    # Multiply from the left
    i = 0
    while i < len(nums):
        output[i] = output[i] * product
        product = product * nums[i]
        i += 1

    product = 1

    # Multiply from the right
    j = len(nums) - 1
    while j >= 0:
        output[j] = output[j] * product
        product = product * nums[j]
        j -= 1

    return output
print(productExceptSelf([1,2,3,4])) #[24,12,8,6]
print(productExceptSelf([2,3,4,5])) #[60, 40, 30, 24]

print("BREAK")

# Leetcode Container With Most Water Problem

def maxArea(height):

    maxArea = 0
    left = 0
    right = len(height) - 1

    while left < right:
        currentArea = min(height[left], height[right]) * (right - left)
        maxArea = max(currentArea, maxArea)
        
        if height[left] < height[right]:
            left =+ 1
        else:
            right -= 1
    
    return maxArea
print(maxArea([1,8,6,2,5,4,8,3,7])) #49

print("break")

# Leetcode Best Time to Buy and Sell Stock

def maxProfit(prices):

    maxProfit = 0
    cheapestPrice = prices[0]

    i = 0
    while i < len(prices):
        price = prices[i]
        if price < cheapestPrice:
            cheapestPrice = price

        currentProfit = price - cheapestPrice
        maxProfit = max(currentProfit, maxProfit)
        i += 1

    return maxProfit

print(maxProfit([7,1,5,3,6,4])) #5
print(maxProfit([7,6,4,3,1])) #0