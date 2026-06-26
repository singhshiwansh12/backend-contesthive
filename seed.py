"""
seed.py - Seed 20 solutions (C++, Python, Java) into ContestHive database.
Run: python seed.py
"""
import os
import json
import sys
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/contesthive")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

genai.configure(api_key=GEMINI_API_KEY)

sys.path.insert(0, os.path.dirname(__file__))
from main import Base, Solution, get_embedding

# FORCE pg8000 driver to avoid psycopg2/C++ build errors on Windows
SAFE_DB_URL = DATABASE_URL.replace("postgresql://", "postgresql+pg8000://")
SAFE_DB_URL = SAFE_DB_URL.replace("postgres://", "postgresql+pg8000://")
SAFE_DB_URL = SAFE_DB_URL.replace("postgresql+psycopg2://", "postgresql+pg8000://")

engine = create_engine(SAFE_DB_URL)
SessionLocal = sessionmaker(bind=engine)

# ─── Seed Data (20 solutions) ─────────────────────────────────────────────────
SOLUTIONS = [
    # ── LeetCode Easy ──────────────────────────────────────────────────────────
    {
        "contest": "Weekly Contest 345",
        "platform": "LeetCode",
        "problem": "Two Sum",
        "difficulty": "Easy",
        "language": "C++",
        "topics": ["Array", "Hash Table"],
        "code": """\
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> mp;
        for (int i = 0; i < nums.size(); i++) {
            int complement = target - nums[i];
            if (mp.find(complement) != mp.end()) {
                return {mp[complement], i};
            }
            mp[nums[i]] = i;
        }
        return {};
    }
};""",
        "explanation": "Hash map stores each number's index. For each element, check if its complement (target - nums[i]) already exists in the map. O(n) time, O(n) space.",
    },
    {
        "contest": "Biweekly Contest 110",
        "platform": "LeetCode",
        "problem": "Valid Parentheses",
        "difficulty": "Easy",
        "language": "C++",
        "topics": ["String", "Stack"],
        "code": """\
class Solution {
public:
    bool isValid(string s) {
        stack<char> st;
        for (char c : s) {
            if (c == '(' || c == '{' || c == '[') {
                st.push(c);
            } else {
                if (st.empty()) return false;
                char top = st.top(); st.pop();
                if (c == ')' && top != '(') return false;
                if (c == '}' && top != '{') return false;
                if (c == ']' && top != '[') return false;
            }
        }
        return st.empty();
    }
};""",
        "explanation": "Use a stack to match opening and closing brackets. Push opening brackets; for closing brackets, check if the stack top matches. O(n) time.",
    },
    {
        "contest": "Weekly Contest 340",
        "platform": "LeetCode",
        "problem": "Best Time to Buy and Sell Stock",
        "difficulty": "Easy",
        "language": "Python",
        "topics": ["Array", "Dynamic Programming", "Greedy"],
        "code": """\
class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        min_price = float('inf')
        max_profit = 0
        for price in prices:
            if price < min_price:
                min_price = price
            elif price - min_price > max_profit:
                max_profit = price - min_price
        return max_profit""",
        "explanation": "Track the minimum price seen so far and the maximum profit possible. Single pass O(n) solution — no need for DP.",
    },
    {
        "contest": "Biweekly Contest 108",
        "platform": "LeetCode",
        "problem": "Merge Two Sorted Lists",
        "difficulty": "Easy",
        "language": "C++",
        "topics": ["Linked List", "Recursion"],
        "code": """\
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode dummy(0);
        ListNode* cur = &dummy;
        while (l1 && l2) {
            if (l1->val <= l2->val) {
                cur->next = l1;
                l1 = l1->next;
            } else {
                cur->next = l2;
                l2 = l2->next;
            }
            cur = cur->next;
        }
        cur->next = l1 ? l1 : l2;
        return dummy.next;
    }
};""",
        "explanation": "Iterative merge using a dummy head node. Compare both lists' current nodes and append the smaller one. Attach remaining nodes at the end. O(m+n) time.",
    },
    {
        "contest": "Weekly Contest 338",
        "platform": "LeetCode",
        "problem": "Climbing Stairs",
        "difficulty": "Easy",
        "language": "Java",
        "topics": ["Dynamic Programming", "Math", "Memoization"],
        "code": """\
class Solution {
    public int climbStairs(int n) {
        if (n <= 2) return n;
        int a = 1, b = 2;
        for (int i = 3; i <= n; i++) {
            int c = a + b;
            a = b;
            b = c;
        }
        return b;
    }
}""",
        "explanation": "Fibonacci-style DP. To reach step n, you come from step n-1 or n-2. Optimized to O(1) space using two variables instead of an array.",
    },

    # ── LeetCode Medium ────────────────────────────────────────────────────────
    {
        "contest": "Weekly Contest 346",
        "platform": "LeetCode",
        "problem": "Longest Substring Without Repeating Characters",
        "difficulty": "Medium",
        "language": "C++",
        "topics": ["String", "Sliding Window", "Hash Table"],
        "code": """\
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_set<char> st;
        int left = 0, maxLen = 0;
        for (int right = 0; right < s.length(); right++) {
            while (st.find(s[right]) != st.end()) {
                st.erase(s[left]);
                left++;
            }
            st.insert(s[right]);
            maxLen = max(maxLen, right - left + 1);
        }
        return maxLen;
    }
};""",
        "explanation": "Sliding window with hash set. Expand right pointer; when duplicate found, shrink from left until duplicate is removed. O(n) time.",
    },
    {
        "contest": "Weekly Contest 350",
        "platform": "LeetCode",
        "problem": "3Sum",
        "difficulty": "Medium",
        "language": "C++",
        "topics": ["Array", "Two Pointers", "Sorting"],
        "code": """\
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<vector<int>> res;
        for (int i = 0; i < nums.size() - 2; i++) {
            if (i > 0 && nums[i] == nums[i-1]) continue;
            int lo = i + 1, hi = nums.size() - 1;
            while (lo < hi) {
                int sum = nums[i] + nums[lo] + nums[hi];
                if (sum == 0) {
                    res.push_back({nums[i], nums[lo], nums[hi]});
                    while (lo < hi && nums[lo] == nums[lo+1]) lo++;
                    while (lo < hi && nums[hi] == nums[hi-1]) hi--;
                    lo++; hi--;
                } else if (sum < 0) lo++;
                else hi--;
            }
        }
        return res;
    }
};""",
        "explanation": "Sort array, then fix one element and use two-pointer approach on the rest. Skip duplicates carefully. O(n²) time, O(1) extra space.",
    },
    {
        "contest": "Weekly Contest 352",
        "platform": "LeetCode",
        "problem": "Coin Change",
        "difficulty": "Medium",
        "language": "Python",
        "topics": ["Dynamic Programming", "BFS", "Array"],
        "code": """\
class Solution:
    def coinChange(self, coins: list[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)
        return dp[amount] if dp[amount] != float('inf') else -1""",
        "explanation": "Bottom-up DP. dp[i] = min coins to make amount i. For each coin, update all reachable amounts. O(amount * len(coins)) time.",
    },
    {
        "contest": "Biweekly Contest 112",
        "platform": "LeetCode",
        "problem": "Number of Islands",
        "difficulty": "Medium",
        "language": "C++",
        "topics": ["Array", "DFS", "BFS", "Union Find", "Matrix"],
        "code": """\
class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        int count = 0;
        int m = grid.size(), n = grid[0].size();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '1') {
                    dfs(grid, i, j);
                    count++;
                }
            }
        }
        return count;
    }
    void dfs(vector<vector<char>>& grid, int i, int j) {
        int m = grid.size(), n = grid[0].size();
        if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] != '1') return;
        grid[i][j] = '0';
        dfs(grid, i+1, j); dfs(grid, i-1, j);
        dfs(grid, i, j+1); dfs(grid, i, j-1);
    }
};""",
        "explanation": "DFS from each unvisited land cell, marking connected cells as visited ('0'). Each DFS call counts as one island. O(m*n) time.",
    },
    {
        "contest": "Weekly Contest 355",
        "platform": "LeetCode",
        "problem": "Longest Palindromic Substring",
        "difficulty": "Medium",
        "language": "C++",
        "topics": ["String", "Dynamic Programming", "Two Pointers"],
        "code": """\
class Solution {
public:
    string longestPalindrome(string s) {
        int n = s.size(), start = 0, maxLen = 1;
        auto expand = [&](int l, int r) {
            while (l >= 0 && r < n && s[l] == s[r]) { l--; r++; }
            if (r - l - 1 > maxLen) {
                maxLen = r - l - 1;
                start = l + 1;
            }
        };
        for (int i = 0; i < n; i++) {
            expand(i, i);     // odd length
            expand(i, i + 1); // even length
        }
        return s.substr(start, maxLen);
    }
};""",
        "explanation": "Expand around each center (for both odd and even length palindromes). Track the longest found. O(n²) time, O(1) space.",
    },
    {
        "contest": "Weekly Contest 358",
        "platform": "LeetCode",
        "problem": "LRU Cache",
        "difficulty": "Medium",
        "language": "C++",
        "topics": ["Hash Table", "Linked List", "Design"],
        "code": """\
class LRUCache {
    int cap;
    list<pair<int,int>> cache;
    unordered_map<int, list<pair<int,int>>::iterator> mp;
public:
    LRUCache(int capacity) : cap(capacity) {}
    int get(int key) {
        if (!mp.count(key)) return -1;
        cache.splice(cache.begin(), cache, mp[key]);
        return mp[key]->second;
    }
    void put(int key, int value) {
        if (mp.count(key)) {
            mp[key]->second = value;
            cache.splice(cache.begin(), cache, mp[key]);
        } else {
            if (cache.size() == cap) {
                mp.erase(cache.back().first);
                cache.pop_back();
            }
            cache.push_front({key, value});
            mp[key] = cache.begin();
        }
    }
};""",
        "explanation": "Doubly linked list + hash map. List maintains order (front = most recent). O(1) get and put using splice to move nodes to front.",
    },
    {
        "contest": "Biweekly Contest 115",
        "platform": "LeetCode",
        "problem": "Word Search",
        "difficulty": "Medium",
        "language": "Python",
        "topics": ["Array", "Backtracking", "DFS", "Matrix"],
        "code": """\
class Solution:
    def exist(self, board: list[list[str]], word: str) -> bool:
        rows, cols = len(board), len(board[0])
        def dfs(r, c, idx):
            if idx == len(word): return True
            if r < 0 or r >= rows or c < 0 or c >= cols: return False
            if board[r][c] != word[idx]: return False
            temp, board[r][c] = board[r][c], '#'
            found = (dfs(r+1,c,idx+1) or dfs(r-1,c,idx+1) or
                     dfs(r,c+1,idx+1) or dfs(r,c-1,idx+1))
            board[r][c] = temp
            return found
        for r in range(rows):
            for c in range(cols):
                if dfs(r, c, 0): return True
        return False""",
        "explanation": "DFS + backtracking. Temporarily mark visited cells with '#'. Restore on backtrack. Try all starting positions. O(m*n*4^L) worst case.",
    },

    # ── LeetCode Hard ──────────────────────────────────────────────────────────
    {
        "contest": "Weekly Contest 360",
        "platform": "LeetCode",
        "problem": "Median of Two Sorted Arrays",
        "difficulty": "Hard",
        "language": "C++",
        "topics": ["Array", "Binary Search", "Divide and Conquer"],
        "code": """\
class Solution {
public:
    double findMedianSortedArrays(vector<int>& A, vector<int>& B) {
        if (A.size() > B.size()) swap(A, B);
        int m = A.size(), n = B.size();
        int lo = 0, hi = m;
        while (lo <= hi) {
            int i = (lo + hi) / 2;
            int j = (m + n + 1) / 2 - i;
            int maxL_A = (i == 0) ? INT_MIN : A[i-1];
            int minR_A = (i == m) ? INT_MAX : A[i];
            int maxL_B = (j == 0) ? INT_MIN : B[j-1];
            int minR_B = (j == n) ? INT_MAX : B[j];
            if (maxL_A <= minR_B && maxL_B <= minR_A) {
                if ((m + n) % 2 == 1)
                    return max(maxL_A, maxL_B);
                return (max(maxL_A, maxL_B) + min(minR_A, minR_B)) / 2.0;
            } else if (maxL_A > minR_B) hi = i - 1;
            else lo = i + 1;
        }
        return 0;
    }
};""",
        "explanation": "Binary search on the smaller array to find the correct partition point where left half elements <= right half elements. O(log(min(m,n))) time.",
    },
    {
        "contest": "Weekly Contest 362",
        "platform": "LeetCode",
        "problem": "Trapping Rain Water",
        "difficulty": "Hard",
        "language": "C++",
        "topics": ["Array", "Two Pointers", "Stack", "Dynamic Programming"],
        "code": """\
class Solution {
public:
    int trap(vector<int>& height) {
        int lo = 0, hi = height.size() - 1;
        int maxL = 0, maxR = 0, water = 0;
        while (lo < hi) {
            if (height[lo] < height[hi]) {
                if (height[lo] >= maxL) maxL = height[lo];
                else water += maxL - height[lo];
                lo++;
            } else {
                if (height[hi] >= maxR) maxR = height[hi];
                else water += maxR - height[hi];
                hi--;
            }
        }
        return water;
    }
};""",
        "explanation": "Two-pointer approach. The side with the smaller height determines the water level. Process the shorter side inward, accumulating trapped water. O(n) time, O(1) space.",
    },

    # ── Codeforces ────────────────────────────────────────────────────────────
    {
        "contest": "Codeforces Round 900 (Div. 2)",
        "platform": "Codeforces",
        "problem": "Sasha and the Beautiful Array",
        "difficulty": "Easy",
        "language": "C++",
        "topics": ["Greedy", "Sorting", "Math"],
        "code": """\
#include <bits/stdc++.h>
using namespace std;
int main() {
    int t;
    cin >> t;
    while (t--) {
        int n;
        cin >> n;
        vector<int> a(n);
        for (auto& x : a) cin >> x;
        sort(a.begin(), a.end());
        long long ans = 0;
        for (int i = 1; i < n; i++)
            ans += a[i] - a[i-1];
        cout << ans << "\\n";
    }
    return 0;
}""",
        "explanation": "Sort the array. The maximum sum of differences between adjacent elements equals a[n-1] - a[0] (telescoping sum). Sorting maximizes the answer.",
    },
    {
        "contest": "Codeforces Round 895 (Div. 3)",
        "platform": "Codeforces",
        "problem": "Imbalanced Arrays",
        "difficulty": "Medium",
        "language": "C++",
        "topics": ["Stack", "Greedy", "Array"],
        "code": """\
#include <bits/stdc++.h>
using namespace std;
int main() {
    int t;
    cin >> t;
    while (t--) {
        int n;
        cin >> n;
        vector<int> a(n);
        for (auto& x : a) cin >> x;
        vector<int> b(n, 0);
        stack<int> st;
        for (int i = 0; i < n; i++) {
            while (!st.empty() && a[st.top()] < a[i])
                st.pop();
            if (!st.empty()) b[i] = b[st.top()] + 1;
            st.push(i);
        }
        for (int x : b) cout << x << " ";
        cout << "\\n";
    }
    return 0;
}""",
        "explanation": "Monotonic stack to find the previous greater or equal element for each position. Build result using the stored counts. O(n) per test case.",
    },
    {
        "contest": "Codeforces Round 888 (Div. 2)",
        "platform": "Codeforces",
        "problem": "Prefix Sums",
        "difficulty": "Easy",
        "language": "Python",
        "topics": ["Math", "Prefix Sum", "Implementation"],
        "code": """\
import sys
input = sys.stdin.readline

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + a[i]
    q = int(input())
    for _ in range(q):
        l, r = map(int, input().split())
        print(prefix[r] - prefix[l-1])

t = int(input())
for _ in range(t):
    solve()""",
        "explanation": "Build prefix sum array in O(n). Each query answered in O(1) using prefix[r] - prefix[l-1]. Total O(n + q) per test case.",
    },

    # ── CodeChef ──────────────────────────────────────────────────────────────
    {
        "contest": "CodeChef Starters 98",
        "platform": "CodeChef",
        "problem": "Binary String MEX",
        "difficulty": "Medium",
        "language": "C++",
        "topics": ["String", "Greedy", "Binary Search"],
        "code": """\
#include <bits/stdc++.h>
using namespace std;
int main() {
    int t;
    cin >> t;
    while (t--) {
        int n;
        string s;
        cin >> n >> s;
        int zeros = count(s.begin(), s.end(), '0');
        int ones  = n - zeros;
        // MEX of subsequence lengths
        int mex = 0;
        while (true) {
            // can we form a string of length mex+1?
            // need at least 1 zero and mex ones, or vice versa
            if (mex + 1 <= ones + zeros) mex++;
            else break;
        }
        cout << mex << "\\n";
    }
    return 0;
}""",
        "explanation": "Count zeros and ones. The MEX is determined by whether we can construct subsequences of each length. Greedy increment until impossible.",
    },
    {
        "contest": "CodeChef Starters 100",
        "platform": "CodeChef",
        "problem": "Chef and Subsequences",
        "difficulty": "Hard",
        "language": "C++",
        "topics": ["Dynamic Programming", "Combinatorics", "Modular Arithmetic"],
        "code": """\
#include <bits/stdc++.h>
using namespace std;
const int MOD = 1e9 + 7;

long long power(long long base, long long exp, long long mod) {
    long long result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = result * base % mod;
        base = base * base % mod;
        exp >>= 1;
    }
    return result;
}

int main() {
    int t;
    cin >> t;
    while (t--) {
        int n, k;
        cin >> n >> k;
        vector<int> a(n);
        for (auto& x : a) cin >> x;
        // Count subsequences with sum divisible by k
        vector<long long> dp(k, 0);
        dp[0] = 1;
        for (int x : a) {
            vector<long long> ndp(k, 0);
            for (int r = 0; r < k; r++) {
                ndp[r] = (ndp[r] + dp[r]) % MOD;
                ndp[(r + x) % k] = (ndp[(r + x) % k] + dp[r]) % MOD;
            }
            dp = ndp;
        }
        cout << (dp[0] - 1 + MOD) % MOD << "\\n";
    }
    return 0;
}""",
        "explanation": "DP where dp[r] = number of subsequences with sum ≡ r (mod k). For each element, either include it or not. Subtract 1 for the empty subsequence. O(n*k) per test case.",
    },

    # ── AtCoder ───────────────────────────────────────────────────────────────
    {
        "contest": "AtCoder Beginner Contest 312",
        "platform": "AtCoder",
        "problem": "Invisible Hand",
        "difficulty": "Medium",
        "language": "C++",
        "topics": ["Binary Search", "Sorting", "Greedy"],
        "code": """\
#include <bits/stdc++.h>
using namespace std;
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    int n, m;
    cin >> n >> m;
    vector<int> a(n), b(m);
    for (auto& x : a) cin >> x;
    for (auto& x : b) cin >> x;
    sort(a.begin(), a.end());
    sort(b.begin(), b.end());
    // Binary search on the price x
    // sellers willing to sell: count of a[i] <= x
    // buyers willing to buy:  count of b[i] >= x
    int lo = 1, hi = 1e9, ans = 1e9;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int sell = upper_bound(a.begin(), a.end(), mid) - a.begin();
        int buy  = b.end() - lower_bound(b.begin(), b.end(), mid);
        if (sell >= buy) { ans = mid; hi = mid - 1; }
        else lo = mid + 1;
    }
    cout << ans << "\\n";
    return 0;
}""",
        "explanation": "Binary search on price. For a given price, count sellers (a[i] <= price) and buyers (b[i] >= price). Find minimum price where supply >= demand. O((n+m) log(maxVal)).",
    },
    {
        "contest": "AtCoder Beginner Contest 315",
        "platform": "AtCoder",
        "problem": "Takahashi's Cake",
        "difficulty": "Easy",
        "language": "Java",
        "topics": ["Math", "Modular Arithmetic", "Implementation"],
        "code": """\
import java.util.*;
public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        long[] a = new long[n];
        for (int i = 0; i < n; i++) a[i] = sc.nextLong();
        long MOD = 998244353;
        long total = 0;
        for (long x : a) total = (total + x) % MOD;
        // Each subset XOR contribution
        long ans = 0;
        for (int bit = 0; bit < 60; bit++) {
            long cnt = 0;
            for (long x : a) if (((x >> bit) & 1) == 1) cnt++;
            long ways = (cnt * (n - cnt)) % MOD;
            long pw = modPow(2, n - 2, MOD);
            ans = (ans + ways % MOD * pw % MOD * (1L << bit)) % MOD;
        }
        System.out.println(ans);
    }
    static long modPow(long b, long e, long mod) {
        long r = 1; b %= mod;
        for (; e > 0; e >>= 1) {
            if ((e & 1) == 1) r = r * b % mod;
            b = b * b % mod;
        }
        return r;
    }
}""",
        "explanation": "For each bit, count numbers with that bit set (cnt) and unset (n-cnt). Each pair contributes to XOR. Multiply by 2^(n-2) for remaining elements' subsets. O(n * 60).",
    },
]


def seed():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        existing = db.query(Solution).count()
        if existing > 0:
            print(f"⚠️  Database already has {existing} solution(s). Skipping seed.")
            print("   To re-seed, first clear the table: DELETE FROM solutions;")
            return

        print(f"🌱 Seeding {len(SOLUTIONS)} solutions...\n")
        for i, s in enumerate(SOLUTIONS, 1):
            embed_text = f"{s['problem']} {s['explanation']} {' '.join(s['topics'])}"
            print(f"  [{i:02d}/{len(SOLUTIONS)}] 🔄 Embedding: {s['problem']} ({s['platform']})")
            embedding = get_embedding(embed_text)

            sol = Solution(
                contest=s["contest"],
                platform=s["platform"],
                problem=s["problem"],
                difficulty=s["difficulty"],
                language=s["language"],
                topics=json.dumps(s["topics"]),
                code=s["code"],
                explanation=s["explanation"],
                embedding=embedding,
            )
            db.add(sol)

        db.commit()
        print(f"\n✅ Successfully seeded {len(SOLUTIONS)} solutions!")
        print("\n📊 Summary:")
        platforms = {}
        for s in SOLUTIONS:
            platforms[s["platform"]] = platforms.get(s["platform"], 0) + 1
        for p, c in sorted(platforms.items()):
            print(f"   {p}: {c} solutions")
    except Exception as e:
        db.rollback()
        print(f"\n❌ Seeding failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed()