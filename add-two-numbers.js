class ListNode{
  constructor(val,next){
      this.val = (val===undefined ? 0 : val)
      this.next = (next===undefined ? null : next)
  }
}

const addTwoNumbers = function(l1, l2) {
  let temp1=l1
  let temp2=l2
  let sum = l1.val + l2.val 
  let initialCarry = Math.floor(sum / 10);
  let digit = sum % 10
  let previousNode = new ListNode(digit, null)
  const headNode=previousNode
  function recursiveAdding(temp1,temp2,carry=0){
      if (temp1===null && temp2===null){
          if(carry===0) return null
          const currentNode= new ListNode(carry,null)
          previousNode.next=currentNode
          previousNode=currentNode
          return null
      }else if (temp2===null){
          temp2=new ListNode(0,null);
      }else if (temp1===null){
          temp1=new ListNode(0,null);
      }
      let sum=temp1.val+temp2.val+carry
      console.log(sum)
      carry=Math.floor(sum/10);
      let digit=sum%10
      const currentNode= new ListNode(digit,null)
      console.log(digit,currentNode)
      previousNode.next=currentNode
      previousNode=currentNode
      return recursiveAdding(temp1.next,temp2.next,carry)
  }
  recursiveAdding(l1.next,l2.next,initialCarry)
  return headNode
 };
 
 // testArray = new ListNode(2,new ListNode(4,new ListNode(3,null)))
 // secondTest=new ListNode(5,new ListNode(6,new ListNode(4,null)))
 // console.log(addTwoNumbers(testArray, secondTest))
 
 testArray = new ListNode(9,new ListNode(9,new ListNode(9,new ListNode(9,null))))
 secondTest=new ListNode(9,new ListNode(9,new ListNode(9,new ListNode(9,new ListNode(9,new ListNode(9,new ListNode(9,null)))))))
 console.log(addTwoNumbers(testArray, secondTest))