#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:44:43 2021

@author: harikavajha
"""

import sys
 
 
 
def maxSumFlip(a, n):
 
   
    total_sum = 0
    for i in range(n):
        total_sum += a[i]
 
    
    max_sum = -sys.maxsize - 1
 
  
    for i in range(n):
 
       
        sum = 0
 
      
        flip_sum = 0
 
        for j in range(i, n):
 
           
            sum += a[j]
 
           
            max_sum = max(max_sum,
                          total_sum - 2 * sum)
 
    # Return the max_sum
    return max(max_sum, total_sum)
 
 
# Driver Code
arr = []
N = len(arr)
 
print(maxSumFlip(arr, N))