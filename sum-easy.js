const twoSum = function (nums, target) {
  let indices
  function checkTwin(n, i) {
    nums.map((m, j) => { m == n && j != i ? indices=[i,j]: null })
  }
  function valueTarget(n,i) {
    let copy = nums.map((m) => m = target - m)
    copy.map((c, j) => {
      // console.log(n,c);
      if (n == c && j !== i) {
        indices = [i, j];
      }
    })
  }
  while (indices == undefined) {
    nums.map((n, i) => n <= target ? (target == n ? checkTwin(n, i) : valueTarget(n, i)) : null)
  }
 return indices
};
let nums = [2, 7, 11, 15]
let target = 9
console.log(twoSum(nums, target))
nums = [3, 3]
target = 6
console.log(twoSum(nums, target))
nums = [0,4, 3,0]
target = 0
console.log(twoSum(nums, target))
nums = [3,2,4]
target = 6
console.log(twoSum(nums, target))
nums = [2,4,11,3]
target = 6
console.log(twoSum(nums, target))
nums = [-2, -4, -11, -3]
target = -6
console.log(twoSum(nums, target))