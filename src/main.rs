use proconio::input;
use std::cell::RefCell;
use std::rc::Rc;

fn sign(x: isize) -> isize {
    if x > 0 {
        1
    } else if x < 0 {
        -1
    } else {
        0
    }
}

fn orientation(a: (usize, usize), b: (usize, usize), c: (usize, usize)) -> isize {
    let ax = a.0 as isize;
    let ay = a.1 as isize;
    let bx = b.0 as isize;
    let by = b.1 as isize;
    let cx = c.0 as isize;
    let cy = c.1 as isize;

    let cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    sign(cross)
}

fn segments_intersect(
    p1: (usize, usize),
    p2: (usize, usize),
    q1: (usize, usize),
    q2: (usize, usize),
) -> bool {
    if p1.0.max(p2.0) < q1.0.min(q2.0)
        || q1.0.max(q2.0) < p1.0.min(p2.0)
        || p1.1.max(p2.1) < q1.1.min(q2.1)
        || q1.1.max(q2.1) < p1.1.min(p2.1)
    {
        return false;
    }

    let o1 = orientation(p1, p2, q1);
    let o2 = orientation(p1, p2, q2);
    let o3 = orientation(q1, q2, p1);
    let o4 = orientation(q1, q2, p2);

    (o1 * o2 <= 0) && (o3 * o4 <= 0)
}

fn dot(p1: &Vec<f64>, p2: &Vec<f64>) -> f64 {
    assert!(p1.len()==p2.len(), "Wrong size p1: {} != p2: {}", p1.len(), p2.len());
    p1.iter().zip(p2.iter()).map(|(a, b)| a * b).sum()
}

fn add(p1: &Vec<f64>, p2: &Vec<f64>) -> Vec<f64> {
    assert!(p1.len()==p2.len(), "Wrong size p1: {} != p2: {}", p1.len(), p2.len());
    p1.iter().zip(p2.iter()).map(|(a, b)| a + b).collect()
}

fn subtract(p1: &Vec<f64>, p2: &Vec<f64>) -> Vec<f64> {
    assert!(p1.len()==p2.len(), "Wrong size p1: {} != p2: {}", p1.len(), p2.len());
    p1.iter().zip(p2.iter()).map(|(a, b)| a - b).collect()
}


struct Input {
    n: usize,
    m: usize,
    k: usize,
    pos: Vec<(usize, usize)>, // 処理装置, 分別器設置場所, 搬入口
    p: Vec<Vec<f64>>, // 分別確率 p[i][j]
}

impl Input {
    /// proconio マクロでまとめて読み込む
    fn parse_input() -> Self {
        input! {
            n: usize,
            m: usize,
            k: usize,
            d_pos: [(usize, usize); n],
            s_pos: [(usize, usize); m],
            p: [[f64; n]; k],
        }
        let mut pos: Vec<(usize, usize)> = d_pos;
        pos.extend(s_pos);
        pos.push((0, 5000)); // 搬入口
        Input { n, m, k, pos, p }
    }
}

type NodeRef = Rc<RefCell<Node>>;

/// ノード構造（処理装置 or 分別器）
struct Node {
    input: Rc<Input>,
    id: usize,
    machine_id: isize,            // 設置する処理装置 or 分別器のID
    parent: Vec<NodeRef>,          // 接続済み親ノードID一覧
    child: [Option<NodeRef>; 2],   // 接続済み子ノードIDスロット
    in_prob: Vec<Vec<f64>>,      // このノードに入力される各ゴミ種別の確率
    out_prob: [Vec<f64>; 2],     // 各スロットからの出力確率
}

impl Node {
    fn create_processor(id: usize, machine_id: isize, input: &Rc<Input>) -> Rc<RefCell<Self>> {
        let n = input.n;
        let mut out_prob = [vec![0.0; n], vec![0.0; n]];
        out_prob[0][id] = 1.0;
        
        Rc::new(RefCell::new(Self {
            input: Rc::clone(input),
            id,
            machine_id,
            parent: Vec::new(),
            child: [None, None],
            in_prob: Vec::new(),
            out_prob,
        }))
    }

    fn create_sorter(id: usize, input: &Rc<Input>) -> Rc<RefCell<Self>> {
        let n = input.n;
        
        Rc::new(RefCell::new(Self {
            input: Rc::clone(input),
            id,
            machine_id: -1,
            parent: Vec::new(),
            child: [None, None],
            in_prob: Vec::new(),
            out_prob: [vec![0.0; n], vec![0.0; n]],
        }))
    }

    fn create_root(input: &Rc<Input>) -> Rc<RefCell<Self>> {
        let n = input.n;
        let id = n + input.m;
        Rc::new(RefCell::new(Self {
            input: Rc::clone(input),
            id,
            machine_id: -1,
            parent: Vec::new(),
            child: [None, None],
            in_prob: vec![vec![1.0; n]],
            out_prob: [vec![0.0; n], vec![0.0; n]],
        }))
    }

    fn set_sorter(&mut self, k: usize) {
        self.machine_id = k as isize;
    }

    fn is_root(&self) -> bool {
        self.id == self.input.n + self.input.m
    }

    fn is_sorter(&self) -> bool {
        !self.is_root() && self.id >= self.input.n
    }
    
    fn is_processor(&self) -> bool {
        !self.is_root() && !self.is_sorter()
    }

    fn standby(&self) -> bool {
        // そのまま接続してよいかどうか
        self.is_processor() || self.has_sorter()
    }

    fn has_sorter(&self) -> bool {
        self.is_sorter() && self.machine_id >= 0 && self.machine_id < self.input.m as isize
    }

    fn out_prob(&self) -> Vec<f64> {
        add(&self.out_prob[0], &self.out_prob[1])
    }

    fn child(&self, i: usize) -> NodeRef {
        Rc::clone(self.child[i].as_ref().unwrap())
    }

    fn child_cnt(&self) -> usize {
        if self.is_root() { 1 } else { 2 }
    }

    fn update_out_prob(&mut self, child_out_prob: &Vec<Vec<f64>>) {
        assert!(child_out_prob.len()==2, "child_out_prob must be 2.");
        self.out_prob[0] = child_out_prob[0].clone();
        self.out_prob[1] = child_out_prob[1].clone();
    }
}

struct Solver {
    input: Rc<Input>,
    root: NodeRef,
    nodes: Vec<NodeRef>,
    edges: Vec<(usize, usize)>,
}

impl Solver {
    fn new(input: &Rc<Input>) -> Self {
        // ノードは処理装置(n個) + 分別器(m個) + 搬入口(1個)
        let mut nodes = Vec::with_capacity(input.n+input.m+1);
        // 処理装置
        for i in 0..input.n {
            nodes.push(Node::create_processor(i, i as isize, input));
        }
        // 分別器
        for j in 0..input.m {
            nodes.push(Node::create_sorter(input.n+j, input));
        }
        // 搬入口
        let root = Node::create_root(input);
        nodes.push(Rc::clone(&root));
        let edges: Vec<(usize, usize)> = Vec::new();
        Solver { input: Rc::clone(input), root, nodes, edges }
    }

    fn solve(&mut self) {
        let n = self.input.n;
        let m = self.input.m;
        let root_id = self.root.borrow().id;
        self.greedy_connect(root_id);
        self.update_nodes();
        eprintln!("root: {}, in_prob: {:?}, out_prob: {:?}", root_id, self.root.borrow().in_prob, self.root.borrow().out_prob);
        eprintln!("node: {}, in_prob: {:?}, out_prob: {:?}", 0, self.nodes[0].borrow().in_prob, self.nodes[0].borrow().out_prob);
        // self.nodes[n].set_sorter(0);
        // self.connect(self.root, Some(n), None);
        // self.connect(n, Some(0), Some(1));
    }

    fn greedy_connect(&mut self, id: usize) {
        let node = Rc::clone(&self.nodes[id]);
        let (child_cnt, in_prob, out_prob) = {
            let borrow_node = node.borrow();
            assert!(borrow_node.is_root() || borrow_node.is_sorter(), "Node must be root or sorter.");
            let child_cnt = borrow_node.child_cnt();
            let in_prob = borrow_node.in_prob.clone();
            let out_prob = borrow_node.out_prob.clone();
            (child_cnt, in_prob, out_prob)
        };

        for i in 0..child_cnt {
            let mut opt_eval = f64::MIN;
            let mut opt_child = Rc::clone(&self.nodes[0]);
            for (child_id, child) in self.nodes.iter().enumerate() {
                if child_id == id { continue; }
                if !self.connectable(&node, child) { continue; }
                let borrow_child = child.borrow();
                let child_out_prob = borrow_child.out_prob();

                let diff_prob = subtract(&out_prob[i], &child_out_prob);
                let eval = dot(&in_prob[i], &diff_prob);
                if opt_eval > eval {
                    opt_eval = eval;
                    opt_child = Rc::clone(child);
                }
            }
            self.connect(&node, &opt_child, i);
        }
    }

    fn connectable(&self, node: &NodeRef, child: &NodeRef) -> bool {
        if !child.borrow().standby() { return false; }
        let node_id = node.borrow().id;
        let child_id = node.borrow().id;
        let p1 = self.input.pos[node_id];
        let p2 = self.input.pos[child_id];
        for &(id1, id2) in self.edges.iter() {
            let q1 = self.input.pos[id1];
            let q2 = self.input.pos[id2];
            if segments_intersect(p1, p2, q1, q2) {
                return false;
            }
        }

        true
    }

    fn connect(&mut self, node: &NodeRef, child: &NodeRef, i: usize) {
        node.borrow_mut().child[i] = Some(Rc::clone(child));
        child.borrow_mut().parent.push(Rc::clone(node));
        let node_id = node.borrow().id;
        let child_id = child.borrow().id;
        self.edges.push((node_id, child_id));
    }

    fn dfs(&mut self, node: &NodeRef, order: &mut Vec<usize>, in_cnt: &mut Vec<usize>) {
        let mut borrow_node = node.borrow_mut();
        let node_id = borrow_node.id;
        in_cnt[node_id] += 1;
        if in_cnt[node_id] == borrow_node.parent.len() {
            order.push(node_id);
        }
        if !borrow_node.is_processor() { 
            // 処理装置でなければ再帰的に out_prob を更新
            let child_cnt = borrow_node.child_cnt();
            for i in 0..child_cnt {
                let child = borrow_node.child(i);
                self.dfs(&child, order, in_cnt);
                borrow_node.out_prob[i] = child.borrow().out_prob();
            }
        }
    }

    fn update_nodes(&mut self) {
        let mut order: Vec<usize> = Vec::new();
        let mut in_cnt: Vec<usize> = vec![0; self.input.n+self.input.m+1];
        let root = Rc::clone(&self.root);
        self.dfs(&root, &mut order, &mut in_cnt);
    }

    fn score(&self) -> usize {
        let score: f64 = 1_000_000_000.0 * (1.0 - self.root.borrow().out_prob[0].iter().sum::<f64>() / self.input.n as f64);

        score.round() as usize
    }

    fn ans(&self) {
        let n = self.input.n;
        for i in 0..n {
            print!("{} ", self.nodes[i].borrow().machine_id);
        }
        println!("\n{}", self.root.borrow().child(0).borrow().id);
        for i in 0..self.input.m {
            let node = &self.nodes[n+i];
            if node.borrow().parent.is_empty() {
                println!("-1");
            } else {
                println!("{} {} {}", node.borrow().machine_id, node.borrow().child(0).borrow().id, node.borrow().child(1).borrow().id);
            }
        }
    }

    fn result(&self) {
        eprintln!("{{ \"n\": {}, \"m\": {}, \"k\": {}, \"score\": {} }}",
            self.input.n, self.input.m, self.input.k, self.score());
    }
}

fn main() {
    let input = Input::parse_input();
    // Solver 初期化・実行
    let mut solver = Solver::new(&Rc::new(input));
    solver.solve();
    solver.ans();
    solver.result();
}
