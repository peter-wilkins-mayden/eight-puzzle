use std::collections::BinaryHeap;
use num::Integer;

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone)]
struct Board {
    tiles: Vec<i32>,
    n: i32,
}

impl Board {
    fn new(tiles: Vec<i32>, n: i32) -> Board {
        Board { tiles, n }
    }

    fn hamming(&self) -> i32 {
        self.tiles.iter()
            .zip(1..)
            .filter(|(&a, b)| a != *b && a != 0)
            .map(|_| 1).sum()
    }
    fn manhatten(&self) -> i32 {
        self.tiles.iter()
            .zip(1..)
            .filter(|(&x, _)| x != 0)
            .map(|(v, i)| {
                let (d, r) = (*v - i).abs().div_rem(&self.n);
                d + r
            }).sum()
    }
    fn neighbours(&self) -> Vec<Vec<i32>> {
        let zero: i32 = self.tiles.iter().position(|&x| x == 0).unwrap() as i32;
        [zero - 1, zero + 1, zero - self.n, zero + self.n].iter()
            .filter(|&&v| v >= 0 && v < (self.n * self.n))
            .map(|&swap| {
                let mut res = self.tiles.clone();
                res.swap(zero as usize, swap as usize);
                res
            })
            .collect()
    }
}

#[derive(Debug, Clone, Eq)]
struct Node {
    board: Board,
    moves: i32,
    prev: Vec<i32>,
}

impl Node {
    fn new(tiles: Vec<i32>, n: i32, moves: i32, prev: Vec<i32>) -> Node {
        let board = Board::new(tiles, n);
        Node { board, moves, prev }
    }
}

use std::cmp::Ordering;

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        (other.board.manhatten() + other.moves).cmp(&(self.board.manhatten() + self.moves))
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some((other.board.manhatten() + other.moves).cmp(&(self.board.manhatten() + self.moves)))
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        (other.board.manhatten() + other.moves) == (self.board.manhatten() + self.moves)
    }
}

struct Solver {
    initial: Vec<i32>,
    goal: Vec<i32>,
    n: i32,
}

impl Solver {
    fn new(initial: Vec<i32>, n: i32) -> Solver {
        let mut goal: Vec<i32> = (1..).take((n * n - 1) as usize).collect();
        goal.push(0);
        Solver { initial, n, goal }
    }
    fn is_solvable(&self) -> bool {
        let mut heap = BinaryHeap::new();
        let mut heap_swapped = BinaryHeap::new();
        heap.push(Node::new(self.initial.clone(), self.n, 0, Vec::new()));

        let is : Vec<usize> = self.initial.iter()
            .enumerate()
            .filter(|(_, &x)| x != 0)
            .take(2)
            .map(|(v, _)| v).collect();

        let mut swapped = self.initial.clone();
        swapped.swap(is[0], is[1]);
        heap_swapped.push(Node::new(swapped, self.n, 0, Vec::new()));
        loop {
            let tiles = self.step(&mut heap);
            let tiles_swapped = self.step(&mut heap_swapped);
            if tiles == self.goal {
                return true;
            }
            if tiles_swapped == self.goal {
                return false;
            }
        }
    }


    fn solution(&self) -> Vec<Vec<i32>> {
        let mut heap = BinaryHeap::new();
        heap.push(Node::new(self.initial.clone(), self.n, 0, Vec::new()));
        let mut res: Vec<Vec<i32>> = Vec::new();

        loop {
            let tiles = self.step(&mut heap);
            res.push(tiles.clone());
            if tiles == self.goal {
                return res;
            }
        }
    }

    fn step(&self, heap: &mut BinaryHeap<Node>) -> Vec<i32> {
        let min_node = heap.pop().unwrap();
        let tiles = min_node.board.tiles.clone();
        let new_moves = min_node.moves + 1;
        let prev = min_node.prev;
        let neighbours = min_node.board.neighbours();

        neighbours.iter()
            .filter(|&neighbour| neighbour != &prev)
            .for_each(|neighbour| {
                let nb: Node = Node::new(neighbour.clone(), self.n, new_moves, tiles.clone());
                heap.push(nb);
            });
        min_node.board.tiles
    }
}

fn main() {
    let s = Solver::new(vec![0, 1, 3, 4, 2, 5, 7, 8, 6], 3);
    assert_eq!(s.solution(),
               vec![vec![0, 1, 3, 4, 2, 5, 7, 8, 6],
                    vec![1, 0, 3, 4, 2, 5, 7, 8, 6],
                    vec![0, 2, 3, 4, 5, 0, 7, 8, 6],
                    vec![0, 1, 3, 4, 2, 6, 7, 8, 0], ])
}

#[test]
fn test_hamming() {
    let b = Board::new(vec![1, 8, 3, 4, 5, 6, 7, 2], 3);
    assert_eq!(2, b.hamming());
    let c = Board::new(vec![1, 2, 3, 4, 5, 6, 7, 8], 3);
    assert_eq!(0, c.hamming());
    let d = Board::new(vec![2, 3, 1, 5, 6, 4, 8, 7], 3);
    assert_eq!(8, d.hamming());
    let d = Board::new(vec![4, 1, 3, 0, 2, 5, 7, 8, 6], 3);
    assert_eq!(5, d.hamming());
    let d = Board::new(vec![1, 0, 3, 4, 2, 5, 7, 8, 6], 3);
    assert_eq!(3, d.hamming());
}

#[test]
fn test_manhatten() {
    let b = Board::new(vec![1, 8, 3, 4, 5, 0, 7, 2, 6], 3);
    assert_eq!(5, b.manhatten());
    let c = Board::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 0], 3);
    assert_eq!(0, c.manhatten());
    let d = Board::new(vec![2, 0, 1, 5, 6, 4, 8, 7, 3], 3);
    assert_eq!(11, d.manhatten());
    let d = Board::new(vec![4, 1, 3, 0, 2, 5, 7, 8, 6], 3);
    assert_eq!(5, d.manhatten());
    let d = Board::new(vec![1, 0, 3, 4, 2, 5, 7, 8, 6], 3);
    assert_eq!(3, d.manhatten());
}

#[test]
fn test_neighbours() {
    let d = Board::new(vec![1, 0, 3, 4, 2, 5, 7, 8, 6], 3);
    assert_eq!(
        vec![
            vec![0, 1, 3, 4, 2, 5, 7, 8, 6, ],
            vec![1, 3, 0, 4, 2, 5, 7, 8, 6, ],
            vec![1, 2, 3, 4, 0, 5, 7, 8, 6, ],
        ],
        d.neighbours());
    let e = Board::new(vec![0, 1, 3, 4, 2, 5, 7, 8, 6], 3);
    assert_eq!(
        vec![
            vec![1, 0, 3, 4, 2, 5, 7, 8, 6, ],
            vec![4, 1, 3, 0, 2, 5, 7, 8, 6],
        ],
        e.neighbours());
}

#[test]
fn test_goal() {
    let s = Solver::new(vec![0, 1, 3, 4, 2, 5, 7, 8, 6], 3);
    assert_eq!(s.goal, vec![1, 2, 3, 4, 5, 6, 7, 8, 0])
}

#[test]
fn test_solution() {
    let s = Solver::new(vec![0, 1, 3, 4, 2, 5, 7, 8, 6], 3);
    assert_eq!(s.solution(),
               vec![vec![0, 1, 3, 4, 2, 5, 7, 8, 6],
                    vec![1, 0, 3, 4, 2, 5, 7, 8, 6],
                    vec![1, 2, 3, 4, 0, 5, 7, 8, 6],
                    vec![1, 2, 3, 4, 5, 0, 7, 8, 6],
                    vec![1, 2, 3, 4, 5, 6, 7, 8, 0], ]);
}


#[test]
fn test_ordering() {
    let less = Node::new(vec![4, 1, 3, 0, 2, 5, 7, 8, 6], 3, 0, Vec::new());
    let more = Node::new(vec![1, 0, 3, 4, 2, 5, 7, 8, 6], 3, 0, Vec::new());
    assert_eq!(Ordering::Greater, more.cmp(&less));
}

#[test]
fn test_heap() {
    let mut heap = BinaryHeap::new();
    let a = Node::new(vec![4, 1, 3, 0, 2, 5, 7, 8, 6], 3, 1, Vec::new());
    let b = Node::new(vec![1, 0, 3, 4, 2, 5, 7, 8, 6], 3, 1, Vec::new());
    heap.push(a.clone());
    heap.push(b.clone());
    assert_eq!(b, heap.pop().unwrap());
}

#[test]
fn test_heap_one_iteration() {
    let mut heap = BinaryHeap::new();
    let a = Node::new(vec![0, 1, 3, 4, 2, 5, 7, 8, 6], 3, 0, Vec::new());
    let expected = Node::new(vec![1, 0, 3, 4, 2, 5, 7, 8, 6], 3, 1, vec![0, 1, 3, 4, 2, 5, 7, 8, 6]);
    heap.push(a.clone());
    let min_node = heap.pop().unwrap();
    assert_eq!(&min_node, &a);
    let tiles = min_node.board.tiles.clone();
    let new_moves = min_node.moves + 1;
    let prev = min_node.prev;
    assert_eq!(&prev, &vec![]);
    let neighbours = min_node.board.neighbours();

    neighbours.iter()
        .filter(|&neighbour| neighbour != &prev)
        .for_each(|neighbour| {
            let nb: Node = Node::new(neighbour.clone(), 3, new_moves, tiles.clone());
            heap.push(nb);
        });
    assert_eq!(heap.pop().unwrap(), expected);
}

#[test]
fn test_is_solvable() {
    let s = Solver::new(vec![0, 1, 3, 4, 2, 5, 7, 8, 6], 3);
    assert!(s.is_solvable());
    let s = Solver::new(vec![1, 2, 3, 4, 5, 6, 8, 7, 0], 3);
    assert!(!s.is_solvable());
}