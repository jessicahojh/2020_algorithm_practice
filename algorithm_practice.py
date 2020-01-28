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

    can_reach = 0
    for idx, num in enumerate(nums):
        # i can't reach idx, then I can't move forward
        if idx > can_reach:
            return False

        can_reach = max(can_reach, idx + num) 
        #I just passed my goal
        if can_reach >= len(nums) - 1:
            return True

    return False

print(canJump2([2,3,1,1,4])) # True
print(canJump2([3,2,1,0,4])) # False

print("BREAK")

def firstMissingPositive(nums):
    # O(n^2) runtime 

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
    # O(n) runtime with extra space used

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

def firstMissingPositive3(nums):
    # O(n) runtime and no extra space used

    result = len(nums) + 1

    i = 0
    while i < len(nums):
        while nums[i] > 0 and nums[i] != nums[nums[i] - 1] and nums[i] < i + 1:
            index = nums[i] - 1
            temp = nums[index]
            nums[index] = nums[i]
            nums[i] = temp
        i+=1

    i = 0
    while i < len(nums):
        if nums[i] != i + 1:
            result = i + 1
            break
        i+=1

    return result

# print(firstMissingPositive3([])) #1
# print(firstMissingPositive3([1,2,0])) #3
# print(firstMissingPositive3([3,4,-1,1])) #2
# print(firstMissingPositive3([7,8,9,11,12])) #1

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

def productExceptSelf2(nums): # my way

    answer = []

    for i, num in enumerate(nums):
        answer.append(multiply(nums[:i]) * multiply(nums[i+1:]))

    return answer

def multiply(lst):
    
    answer = 1

    for num in lst:
        answer = num * answer

    return answer


print(productExceptSelf2([1,2,3,4])) #[24,12,8,6]
print(productExceptSelf2([2,3,4,5])) #[60, 40, 30, 24]

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
            left += 1
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

print("BREAK")

# This problem was asked by Facebook.
# Given the mapping a = 1, b = 2, ... z = 26, and an encoded message,
# count the number of ways it can be decoded.
# For example, the message '111' would give 3, since it could be 
# decoded as 'aaa', 'ka', and 'ak'.
# You can assume that the messages are decodable. For example, '001' 
# is not allowed.

def helper(data, k, memo):

    if k == 0:
        return 1

    s = len(data) - k 

    if k in memo.keys():
        return memo[k]

    result = helper(data, k-1, memo)

    if k >= 2 and int(data[s:s+2]) <= 26:
        result += helper(data, k-2, memo)
    memo[k] = result

    return result

def numWays(s):
    memo = {}
    return helper(s, len(s), memo)

print(numWays('11')) # 2 b/c 'aa', 'k'
# print(numWays('111')) # 3 b/c 'aaa', 'ka', 'ak'
# print(numWays('1111')) # 5 b/c 'aaaa', 'kk', 'aak', 'kaa', 'aka'

print('BREAK')

# without using memoization

def helper2(s, k):

    if k == 0:
        return 1

    x = len(s) - k 

    result = helper2(s, k-1)

    if k >= 2 and int(s[x:x+2]) <= 26:
        result += helper2(s, k-2)

    return result

def numWays2(string):

    return helper2(string, len(string)) 

print(numWays2('11')) # 2 b/c 'aa', 'k'

print("BREAK")

# class TreeNode(object):

#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# class Solution(object):

#     def helperFunction(self, node, counter):

#         if node == None:
#             return
#         self.helperFunction(node.left, counter)
#         if node.left


def countAndSay(n):

    seq = "1"
    for i in range(n-1):
        seq = getNext(seq)
    return seq

def getNext(seq):
    i = 0
    nextSeq = ""

    while i < len(seq):
        count = 1
        while i < len(seq) - 1 and seq[i] == seq [i+1]:
            count += 1
            i += 1
        nextSeq += str(count) + seq[i]
        i += 1
    return nextSeq


print(countAndSay(4))
#1
#11
#21
#1211

print("BREAK")

def MergeTwoSortedLists(L1, L2, x):

    L1Pointer = x
    L2Pointer = len(L2) - 1
    i = len(L1) - 1

    while L2Pointer >= 0 and L1Pointer >=0:
        if L2[L2Pointer] > L1[L1Pointer]:
            L1[i] = L2[L2Pointer]
            L2Pointer -= 1
            i -= 1
        else:
            L1[i] = L1[L1Pointer]
            L1Pointer -= 1
            i -= 1

    while i >= 0:
        if L2Pointer >= 0:
            L1[i] = L2[L2Pointer]
            L2Pointer -= 1
            i -= 1

        elif L1Pointer >= 0:
            L1[i] = L1[L1Pointer]
            L1Pointer -= 1
            i -= 1
    
    return L1
print(MergeTwoSortedLists([4,8,9,12,0,0,0,0], [3,7,10,14], 3))
print(MergeTwoSortedLists([8,9,12,15,20,0,0,0,0], [3,5,7,10], 4))

print("BREAK")

# keeptruckin 1 hour pair

def zipList(lst):

    answer = [] 
    theres_next = True
    i = 0

    while theres_next:

        theres_next = False

        for inside_lst in lst:
            print("we are in list", inside_lst)
            print("and i is ", i)
            print("theres next is", theres_next)
            if i+1 <= len(inside_lst): 
                theres_next = True
                print("there's a next!")
                answer.append(inside_lst[i])
        i += 1

    return answer

print(zipList([[4,15,3,7], [3,7,5], [10], [5,2,16,9]]))

#[
# [4,15,3,7], 
# [3,7,5], 
# [10], 
# [5,2,16,9]
# ]

#[4,3,10,5,15,7,2,3,5,16,7,9]

# note: line 15 needs the i+1 instead of just i, or else we loop 5 times instead 
# of just 4 times


print("BREAK")

# This problem was asked by Airbnb.
# Given a list of integers, write a function that returns the largest sum of 
# non-adjacent numbers. Numbers can be 0 or negative.
# For example, [2, 4, 6, 2, 5] should return 13, since we pick 2, 6, and 5. 
# [5, 1, 1, 5] should return 10, since we pick 5 and 5.
# Follow-up: Can you do this in O(N) time and constant space?

def largestSumNonAdjacent(nums):

    if len(nums) == 0:
        return 0
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums[0], nums[1])

    answer = [nums[0], max(nums[0], nums[1])]

    i = 2

    while i < len(nums):
        answer.append(max(nums[i] + answer[i-2], answer[i-1]))
        i += 1

    return answer.pop()

print(largestSumNonAdjacent([2, 4, 6, 2, 5])) # 13
print(largestSumNonAdjacent([5, 1, 1, 5])) # 10

print("BREAK")

# now redo largestSumNonAdjacent with constant space

def constantSpaceLargestSumNonAdjacent(nums):

    if len(nums) == 0:
        return 0
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums[0], nums[1])

    nums[1] = max(nums[0], nums[1])

    i = 2

    while i < len(nums):
        nums[i] = max(nums[i] + nums[i-2], nums[i-1])
        i += 1

    return nums[-1]

print(constantSpaceLargestSumNonAdjacent([2, 4, 6, 2, 5])) # 13
print(constantSpaceLargestSumNonAdjacent([5, 1, 1, 5])) # 10

print("BREAK")

def romanToNumeral(roman):

    dictionary = {"I":1, "V":5, "X":10, "L":50, "C":100, "D":500, "M":1000}

    answer = 0
    prev = 0
    i = len(roman) - 1

    while i >= 0:
        if dictionary[roman[i]] >= prev:
            answer += dictionary[roman[i]]
            prev = dictionary[roman[i]]
        else:
            answer -= dictionary[roman[i]]
            prev = dictionary[roman[i]]
        i -= 1

    return answer

print(romanToNumeral("XXI")) #21
print(romanToNumeral("IV")) #4
print(romanToNumeral("CDLXXXVI")) #486

print("BREAK")

# def numeralToRoman(num):
#     """num is a int between 1-3999"""

#     dictionary = {"I":1, "IV":4, "V":5, "IX":9, "X":10, "XL":40, "L":50, 
#                   "XC":90, "C":100, "CD":400, "D":500, "CM":900, "M":1000}

#     num = str(num)
#     i = 0

#     answer = []

#     while i < len()


# print(numeralToRoman(44)) # XLIV
# print(numeralToRoman(90)) # XC
# print(numeralToRoman(649)) # DCXLIX

print("BREAK")

def climbStairs(n):

    if n < 1:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2

    return climbStairs(n-1) + climbStairs(n-2)

print(climbStairs(1))
print(climbStairs(2))
print(climbStairs(5))
print(climbStairs(8))

print("BREAK")

def climbStairsWithMemo(n):
    
    memo = {}
    
    return climbStairsHelper(n, memo)

def climbStairsHelper(n, memo):
    if n < 1:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2
    
    if n in memo:
        return memo[n]
    else:    
        memo[n] = climbStairsHelper(n - 1, memo) + climbStairsHelper(n - 2, memo)
        
    return memo[n]

print(climbStairsWithMemo(1))
print(climbStairsWithMemo(2))
print(climbStairsWithMemo(5))
print(climbStairsWithMemo(8))

print("BREAK")

# This problem was asked by Apple.
# Implement a job scheduler which takes in a function f and an integer n, 
# and calls f after n milliseconds.

from threading import Timer

def jobScheduler(f, n):

    return(Timer(n, f))

def f():

    print("Function F got called")

print(jobScheduler(f(), 3.0))

# I also did a version of this in JS using setTimeout

print("BREAK")

def multiplyStrings(num1, num2):

    dictionary = {"0":0, "1":1, "2":2, "3":3, "4":4, 
                  "5":5, "6":6, "7":7, "8":8, "9":9}

    if len(num2) >= len(num1):
        longerNum = num2
        shorterNum = num1
    else:
        longerNum = num1
        shorterNum = num2

    holder = []

    for x in range(len(num1) + len(num2)):
        holder.append(0)

    x = len(holder) - 1
    y = len(holder) - 1
    i = len(shorterNum) - 1
    
    extra = 0

    while i >= 0:
        j = len(longerNum) - 1
        y = x
        while j >= 0:
            if j == 0:
                tempAnswer = dictionary[shorterNum[i]] * dictionary[longerNum[j]] + extra + holder[y]
                
                if tempAnswer > 9:
                    holder[y] = tempAnswer % 10
                    holder[y-1] = tempAnswer//10
                    extra = 0

                else:
                    holder[y] = tempAnswer
                    extra = 0
            else:
                tempAnswer = dictionary[shorterNum[i]] * dictionary[longerNum[j]] + extra + holder[y]
                if tempAnswer > 9:
                    holder[y] = tempAnswer % 10
                    extra = tempAnswer//10
                else:
                    holder[y] = tempAnswer
                    extra = 0
            j -= 1
            y -= 1
        i -= 1
        x -= 1

    print(holder)
    # listToString = ''.join(holder)
    # print(listToString)
    # stringToInt = int(listToString)

    # return stringToInt

print(multiplyStrings("123", "456"))
print(multiplyStrings("271", "387"))

print("BREAK")

# This problem was asked by Twitter.
# Implement an autocomplete system. That is, given a query string s and a 
# set of all possible query strings, return all strings in the set that have 
# s as a prefix.
# For example, given the query string de and the set of strings [dog, deer, 
# deal], return [deer, deal].
# Hint: Try preprocessing the dictionary into a more efficient data structure 
# to speed up queries.

def autocompleteSystem(s, a):

    dictionary = {}

    for word in a:
        for i in range(len(word)-1):
            if word[:i+1] not in dictionary.keys():
                dictionary[word[:i+1]] = [word[i+1:]]
            elif word[:i+1] in dictionary.keys():
                dictionary[word[:i+1]].append(word[i+1:]) 

    answer = []

    if s in dictionary.keys():
        for x in dictionary[s]:
            answer.append(s+x)

    return answer

print(autocompleteSystem('de', ['dog', 'deer', 'deal'])) # ['deer', 'deal']

print("BREAK")

# def IPAddresses(d):

# print(IPAddresses("25525511135")) # ["255.255.11.135", "255.255.111.35"]

print("BREAK")

# https://leetcode.com/problems/camelcase-matching/

# def camelCaseMatching(queries, pattern):

#     upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

#     holder = []

#     answer = []



# print(camelCaseMatching(["FooBar","FooBarTest","FootBall","FrameBuffer","ForceFeedBack"], "FB"))
# [true,false,true,true,false]

print("BREAK")

# def climbStairs2(n):
#     """Instead of being able to climb 1 or 2 steps at a time, you could 
#     climb any number from a set of positive integers X? For example, if 
#     X = {1, 3, 5}, you could climb 1, 3, or 5 steps at a time"""

#     if n < 1:
#         return 0
#     if n == 1:
#         return 1
#     if n == 3:
#         return 2
#     if n == 5:
#         return 3

#     return climbStairs2(n-5) + climbStairs2(n-3) + climbStairs2(n-1)

# print(climbStairs2(10, [1,3,5]))
# print(climbStairs2(10))

# 1

# 11

# 3
# 111

# 1111
# 31
# 13

# 5
# 113
# 11111

# (4, 3) ---> 2
# 0, 0, 1, 1, 2, 4, 7, 13 

def kibb(n, k):

    if n < k - 1:
        return 0

    elif n == 2:
        return 1

    else:
        return sum(kibb(n-x, k) for x in range(1, k + 1))

print(kibb(0, 3))
print(kibb(1, 3))
print(kibb(2, 3))
print(kibb(3, 3))
print(kibb(4, 3))

print("BREAK")

def canAttendMeetings(intervals):

    starts = []
    ends = []

    i = 0

    while i < len(intervals):
        subArray = intervals[i]
        starts.append(subArray[0])
        ends.append(subArray[1])
        i += 1

    starts.sort()
    ends.sort()

    # start [0,5,15]
    # end [10,20,30]

    # start [2,7]
    # end [4,10]

    i = 0

    while i < len(starts):
        if starts[i+1] < ends[i]:
            return False

        return True

print(canAttendMeetings([[0,30],[5,10],[15,20]])) # False
print(canAttendMeetings([[7,10],[2,4]])) # True

print("BREAK")

# https://leetcode.com/problems/repeated-dna-sequences/


def findRepeatedDnaSequences(s):
    
    dictionary = {}
    
    i = 0
    
    while i < len(s) - 10:
        string = s[i:i+10]
        if string not in dictionary:
            dictionary[string] = 1
        else:
            dictionary[string] += 1
        i += 1
            
    answer = []
    
    for key, value in dictionary.items():
        if value > 1:
            answer.append(key)

    return answer
        
print(findRepeatedDnaSequences("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"))
# ["AAAAACCCCC", "CCCCCAAAAA"]

print("BREAK")

# reverse vowels of a string

def reverseVowels(s):

    vowel = {'a','e','i','o','u','A','E','I','O','U'}

    x = list(s)

    i = 0
    j = len(x)-1

    holder = ''

    while i < j:
        if x[i] in vowel and x[j] in vowel:
            holder = x[i]
            x[i] = x[j]
            x[j] = holder
            i += 1
            j -= 1
        elif x[i] in vowel and x[j] not in vowel:
            j -= 1
        elif x[j] in vowel and x[i] not in vowel:
            i += 1
        else:
            i += 1
            j -= 1

    return ''.join(x)

print(reverseVowels('hello')) #holle
print(reverseVowels('leetcode')) #leotcede

print("BREAK")

def moveZeros(nums):

    index = 0

    i = 0

    while i < len(nums):
        num = nums[i]

        if num != 0:
            nums[index] = num
            index += 1
        i += 1

        print(nums)

    i = index

    while i < len(nums):
        nums[i] = 0
        i += 1
        print(nums)

    return nums

print(moveZeros([0,4,12,0,6])) #[4,12,6,0,0]

print("BREAK")

# This problem was asked by Microsoft.
# Given a dictionary of words and a string made up of those words (no spaces), return the original 
# sentence in a list. If there is more than one possible reconstruction, return any of them. If 
# there is no possible reconstruction, then return null.
# For example, given the set of words 'quick', 'brown', 'the', 'fox', and the string "thequickbrownfox", 
# you should return ['the', 'quick', 'brown', 'fox'].
# Given the set of words 'bed', 'bath', 'bedbath', 'and', 'beyond', and the string "bedbathandbeyond", 
# return either ['bed', 'bath', 'and', 'beyond] or ['bedbath', 'and', 'beyond'].

def origSentence(words, s):

    answer = []
    dictionary = {}

    for word in words:
        if word[0] not in dictionary:
            dictionary[word[0]] = [word]
        else:
            dictionary[word[0]].append(word)

    for i, letter in enumerate(s):
        if letter in dictionary:
            for word in dictionary[letter]:
                if s[i:i+len(word)] == word:
                    answer.append(word)

    return answer

print(origSentence(['quick', 'brown', 'the', 'fox'], "thequickbrownfox")) #['the', 'quick', 'brown', 'fox']
print(origSentence(['bed', 'bath', 'bedbath', 'and', 'beyond'], "bedbathandbeyond"))
# ['bed', 'bath', 'and', 'beyond] or ['bedbath', 'and', 'beyond']