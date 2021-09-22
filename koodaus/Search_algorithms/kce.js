// K closest 


const findClosestElements = (arr, k, x) => {
  const p = findClosestElement(arr, x);

  let i = p - 1; // Because we prefer smaller value if there is a tie
  let j = p;

  while (k-- > 0) {
    if (i < 0 || (j < arr.length && Math.abs(arr[j] - x) < Math.abs(arr[i] - x))) {
      // Here we use < instead of <=, beacuse we prefer smaller value if there is a tie
      j++;
    } else {
      i--;
    }
  }

  return arr.slice(i + 1, j);
};

const findClosestElement = (arr, x) => {
  let lo = 0;
  let hi = arr.length - 1;

  while (lo <= hi) {
    const mid = lo + Math.floor((hi - lo) / 2);

    if (arr[mid] === x) {
      return mid;
    }

    if (arr[mid] > x) {
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }

  return lo;
};

export { findClosestElements };