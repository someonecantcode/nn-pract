public class testing {

    public static void main(String[] args){
        System.out.println(maxProfit(new int[] {1, 2}));
    }

    public static int maxProfit(int[] prices) {
        if (prices.length == 1) {
            return 0;
        }

        int p0 = 0;
        int p1 = 1;

        int max = 0;
        // alg. keep moving the first pointer until smallest value. 
        while(p1 < prices.length-1) {
            if(prices[p1] < prices[p0]) {
                p0++;
            }

            System.out.println(prices[p1]);
            System.out.println(prices[p0]);
            max = Math.max(max, prices[p1]-prices[p0]);
            p1++;
        }
        return max;
    }
}