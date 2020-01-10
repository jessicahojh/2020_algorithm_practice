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