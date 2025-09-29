use proconio::input;
use std::cell::RefCell;
use std::rc::Rc;
use rand::{rngs::StdRng, SeedableRng, Rng};
use rand::seq::SliceRandom;
use std::collections::{BinaryHeap, HashSet, HashMap};
use std::cmp::Reverse;
use std::time::{Instant, Duration};
use std::cmp::Ordering;

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
    if p1 == q1 || p1 == q2 || p2 == q1 || p2 == q2 { return false; }
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

fn hadamard(p1: &Vec<f64>, p2: &Vec<f64>) -> Vec<f64> {
    assert!(p1.len() == p2.len(), "Wrong size p1: {} != p2: {}", p1.len(), p2.len());
    p1.iter().zip(p2.iter()).map(|(a, b)| a * b).collect()
}

fn midpoint(p1: (usize, usize), p2: (usize, usize)) -> (usize, usize) {
    let mid_x = (p1.0 + p2.0) / 2;
    let mid_y = (p1.1 + p2.1) / 2;
    (mid_x, mid_y)
}

fn dist(p1: (usize, usize), p2: (usize, usize)) -> usize {
    let dx = p1.0.abs_diff(p2.0);
    let dy = p1.1.abs_diff(p2.1);

    ((dx*dx + dy*dy) as f64).sqrt() as usize
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
        out_prob[0][machine_id as usize] = 1.0;
        
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

    fn has_sorter(&self) -> bool {
        self.is_sorter() && self.machine_id >= 0 && self.machine_id < self.input.k as isize
    }

    fn out_prob(&self) -> Vec<f64> {
        add(&self.out_prob[0], &self.out_prob[1])
    }

    fn in_prob(&self) -> Vec<f64> {
        let mut in_prob = vec![0.0; self.input.n];
        for p in self.in_prob.iter() {
            in_prob = add(&in_prob, p);
        }

        in_prob
    }

    fn child(&self, i: usize) -> NodeRef {
        Rc::clone(self.child[i].as_ref().unwrap())
    }

    fn child_cnt(&self) -> usize {
        if self.is_root() { 1 } else { 2 }
    }

    fn p(&self, child_i: usize) -> Vec<f64> {
        assert!(self.is_root() || self.is_sorter(), "proccessor doesn't have p.");
        let pi = if self.is_sorter() {
            self.input.p[self.machine_id as usize].clone()
        } else {
            vec![1.0; self.input.n]
        };
        if child_i == 0 {
            pi
        } else {
            subtract(&vec![1.0; self.input.n], &pi)
        }
    }

    fn update_out_prob(&mut self, child_out_prob: &Vec<Vec<f64>>) {
        assert!(child_out_prob.len()==2, "child_out_prob must be 2.");
        self.out_prob[0] = child_out_prob[0].clone();
        self.out_prob[1] = child_out_prob[1].clone();
    }
}

#[derive(Clone)]
struct Solver {
    input: Rc<Input>,
    root: NodeRef,
    nodes: Vec<NodeRef>,
    used: Vec<usize>,
    edges: Vec<(usize, usize)>,
    rng: StdRng,
    timer: Instant,
    on_time: u128,
}

impl Solver {
    fn new(input: &Rc<Input>, seed: u64, on_time: u128) -> Self {
        // ノードは処理装置(n個) + 分別器(m個) + 搬入口(1個)
        let mut nodes = Vec::with_capacity(input.n+input.m+1);
        let mut used: Vec<usize> = Vec::new();
        // 乱数器
        let mut rng = StdRng::seed_from_u64(seed);
        // 処理装置
        let mut range: Vec<usize> = (0..input.n).collect();
        range.shuffle(&mut rng);
        for i in 0..input.n {
            nodes.push(Node::create_processor(i, range[i] as isize, input));
            used.push(i);
        }
        // 分別器
        for j in 0..input.m {
            nodes.push(Node::create_sorter(input.n+j, input));
        }
        // 搬入口
        let root = Node::create_root(input);
        used.push(root.borrow().id);
        nodes.push(Rc::clone(&root));
        let edges: Vec<(usize, usize)> = Vec::new();

        // タイマー
        let timer = Instant::now();
        Solver { input: Rc::clone(input), root, nodes, used, edges, rng, timer, on_time }
    }

    fn elapsed_ms(&self) -> u128 {
        self.timer.elapsed().as_millis()
    }

    fn on_time(&self) -> bool {
        self.elapsed_ms() < self.on_time
    }

    fn solve2(&mut self) {
        let root_id = self.root.borrow().id;
        let child_id = self.input.n;
        self.setup_sorter(child_id, 0);
        self.connect(root_id, child_id, 0);
        self.used.push(child_id);
        let ki = self.nodes[self.input.n].borrow().machine_id as usize;
        let first_row = self.input.p[ki].clone();
        let imax = first_row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal)).unwrap().0;
        let imin = first_row.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal)).unwrap().0;
        let mut machine: HashMap<usize, usize> = HashMap::new();
        for i in 0..self.input.n {
            machine.insert(self.nodes[i].borrow().machine_id as usize, i);
        }
        self.connect(child_id, *machine.get(&imax).unwrap(), 0);
        self.connect(child_id, *machine.get(&imin).unwrap(), 1);
        self.update_nodes();
    }

    fn solve(&mut self) -> Self {
        let root_id = self.root.borrow().id;
        self.connect(root_id, 0, 0);
        self.update_nodes();
        let mut trial_cnt = 0;
        let mut trial2_cnt = 0;
        let mut update_cnt = 0;
        let mut opt_score = 1_000_000_000;
        let mut opt_solver = self.clone();
        while self.on_time() {
            trial_cnt += 1;
            let (node_id, child_id, child_i) = self.random_edge();
            let (next_node_id, next_child_id, next_child_i, ki) = self.split_edges(node_id, child_id, child_i);
            if ki < 0 { continue }
            if !self.on_time() { break; }
            let ki = ki as usize;
            self.disconnect(node_id, child_id);
            self.setup_sorter(next_node_id, ki);
            self.used.push(next_node_id);
            self.connect(node_id, next_node_id, child_i);
            self.connect(next_node_id, child_id, (next_child_i+1)%2);
            self.connect(next_node_id, next_child_id, next_child_i);
            self.update_nodes();
            let score = self.score();
            trial2_cnt += 1;
            if score < opt_score {
                eprintln!("score: {} => {}", opt_score, score);
                opt_score = score;
                opt_solver = self.clone();
                update_cnt += 1;
            } else {
                *self = opt_solver.clone();
            }
        }
        eprintln!("trial_cnt: {}, update_cnt: {}, update_rate: {}%", trial_cnt, update_cnt, ((10000*update_cnt) as f64 / trial_cnt as f64).round()/100.0);
        eprintln!("trial2_cnt: {}, update_cnt: {}, update_rate: {}%", trial2_cnt, update_cnt, ((10000*update_cnt) as f64 / trial2_cnt as f64).round()/100.0);

        // opt_solver
        self.clone()
    }

    fn split_edges(&mut self, node_id: usize, child_id: usize, child_i: usize) -> (usize, usize, usize, isize) {
        let max_cnt = 10;
        let pos1 = midpoint(self.input.pos[node_id], self.input.pos[child_id]);
        let mut used_nodes: BinaryHeap<(Reverse<usize>, usize)> = BinaryHeap::new();
        let mut unused_nodes: BinaryHeap<(Reverse<usize>, usize)> = BinaryHeap::new();
        let used: HashSet<usize> = self.used.clone().into_iter().collect::<HashSet<usize>>();
        for (id, &pos2) in self.input.pos.iter().enumerate() {
            if id == node_id || id == child_id { continue; }
            if used.contains(&id) {
                used_nodes.push((Reverse(dist(pos1, pos2)), id));
            } else {
                unused_nodes.push((Reverse(dist(pos1, pos2)), id));
            }
        }

        let node_p = self.nodes[node_id].borrow().p(child_i);
        let child_out_prob = self.nodes[child_id].borrow().out_prob();
        let mut opt_eval = dot(&node_p, &child_out_prob);
        // let mut opt_eval = 0.0;
        eprintln!("opt_eval: {}", opt_eval);
        let mut opt_next_id = node_id;
        let mut opt_next_child_id = child_id;
        let mut opt_next_child_i = child_i;
        let mut opt_ki = -1;
        let mut unused_cnt = 0;
        let mut used_cnt = 0;
        let mut eval_cnt = 0;
        while let Some((_, next_id)) = unused_nodes.pop() {
            unused_cnt += 1;
            // if unused_cnt > max_cnt { break; }
            if !self.connectable(next_id, node_id) { continue; }
            if !self.connectable(next_id, child_id) { continue; }
            while let Some((_, next_child_id)) = used_nodes.pop() {
                used_cnt += 1;
                // if used_cnt > max_cnt { break; }
                if !self.connectable(next_id, next_child_id) { continue; }
                if self.is_connect(next_child_id, node_id) { continue; }
                eval_cnt += 1;
                // if eval_cnt > max_cnt { break; }

                // 接続できるノードが見つかったので、最適な分別器を探索
                let next_child_out_prob = self.nodes[next_child_id].borrow().out_prob();
                for ki in 0..self.input.k {
                    let next_p0 = self.input.p[ki].clone();
                    let next_p1 = subtract(&vec![1.0; self.input.n], &next_p0);

                    let next_child_i = 1;  // 元々の子は 0 に繋げる
                    let child0_prob = hadamard(&next_p0, &child_out_prob);
                    let child1_prob = hadamard(&next_p1, &next_child_out_prob);
                    let eval = dot(&node_p, &child0_prob) + dot(&node_p, &child1_prob);
                    if eval > opt_eval {
                        opt_eval = eval;
                        opt_next_id = next_id;
                        opt_next_child_id = next_child_id;
                        opt_next_child_i = next_child_i;
                        opt_ki = ki as isize;
                    }


                    let next_child_i = 0;  // 元々の子は 1 に繋げる
                    let child0_prob = hadamard(&next_p0, &next_child_out_prob);
                    let child1_prob = hadamard(&next_p1, &child_out_prob);
                    let eval = dot(&node_p, &child0_prob) + dot(&node_p, &child1_prob);
                    if eval > opt_eval {
                        opt_eval = eval;
                        opt_next_id = next_id;
                        opt_next_child_id = next_child_id;
                        opt_next_child_i = next_child_i;
                        opt_ki = ki as isize;
                    }
                if !self.on_time() { break; }
                }
            if !self.on_time() { break; }
            }
        }

        eprintln!("unused: {}, used: {}, eval: {}, ope_eval: {}", unused_cnt, used_cnt, eval_cnt, opt_eval);
        (opt_next_id, opt_next_child_id, opt_next_child_i, opt_ki)
    }

    fn random_edge(&mut self) -> (usize, usize, usize) {
        let random_i = self.rng.gen_range(0..self.edges.len());
        let (node_id, child_id) = self.edges[random_i];
        let node = self.nodes[node_id].borrow();
        let child_i = if node.child(0).borrow().id == child_id { 0 } else { 1 };

        (node_id, child_id, child_i)
    }

    fn connectable(&self, node_id: usize, child_id: usize) -> bool {
        let p1 = self.input.pos[node_id];
        let p2 = self.input.pos[child_id];
        for &(id1, id2) in self.edges.iter() {
            let q1 = self.input.pos[id1];
            let q2 = self.input.pos[id2];
            if segments_intersect(p1, p2, q1, q2) {
                return false;
            }
            if !self.on_time() { break; }
        }

        true
    }

    fn is_connect(&self, frm: usize, to: usize) -> bool {
        if !self.on_time() { return true; }
        // eprintln!("from: {}, to: {}", frm, to);
        let node = self.nodes[frm].borrow();
        if node.child[0].is_none() && node.child[1].is_none() { return false; }
        for child_i in 0..node.child_cnt() {
            let child_id = node.child(child_i).borrow().id;
            if child_id == to { return true; }
            if self.is_connect(child_id, to) { return true; }
        }

        false
    }

    fn connect(&mut self, node_id: usize, child_id: usize, i: usize) {
        // 親と子の接続ノードを更新。in_probは初期値を追加
        let node = Rc::clone(&self.nodes[node_id]);
        let child = Rc::clone(&self.nodes[child_id]);
        node.borrow_mut().child[i] = Some(Rc::clone(&child));
        child.borrow_mut().parent.push(Rc::clone(&node));
        child.borrow_mut().in_prob.push(vec![0.0; self.input.n]);
        let node_id = node.borrow().id;
        let child_id = child.borrow().id;
        self.edges.push((node_id, child_id));
    }

    fn disconnect(&mut self, node_id: usize, child_id: usize) {
        // 子側の親の接続(親ノードとin_prob)
        let child = Rc::clone(&self.nodes[child_id]);
        let parents = child.borrow().parent.clone();
        for (i, node) in parents.iter().enumerate() {
            if node.borrow().id == node_id {
                child.borrow_mut().parent.remove(i);
                child.borrow_mut().in_prob.remove(i);
                break;
            }
        }

        // edgesの削除
        let pos_i = self.edges.iter().position(|&edge| edge == (node_id, child_id)).unwrap();
        self.edges.remove(pos_i);
    }

    fn setup_sorter(&mut self, node_id: usize, ki: usize) {
        let mut node = self.nodes[node_id].borrow_mut();
        node.set_sorter(ki);
    }

    fn dfs(&mut self, node: &NodeRef, order: &mut Vec<usize>, in_cnt: &mut Vec<usize>) {
        let node_id = node.borrow().id;
        in_cnt[node_id] += 1;
        if in_cnt[node_id] == node.borrow().parent.len() {
            order.push(node_id);
        }
        let is_processor = node.borrow().is_processor();
        if !is_processor { 
            // 処理装置でなければ再帰的に out_prob を更新
            let child_cnt = node.borrow().child_cnt();
            for i in 0..child_cnt {
                let child = node.borrow().child(i);
                self.dfs(&child, order, in_cnt);
                let p = node.borrow().p(i);
                node.borrow_mut().out_prob[i] = hadamard(&p, &child.borrow().out_prob());
            }
        }
    }

    fn update_nodes(&mut self) {
        let mut order: Vec<usize> = Vec::new();
        let mut in_cnt: Vec<usize> = vec![0; self.input.n+self.input.m+1];
        let root = Rc::clone(&self.root);

        // out_probの更新
        self.dfs(&root, &mut order, &mut in_cnt);

        // in_probの更新
        for &id in order.iter() {
            let node = Rc::clone(&self.nodes[id]);
            let parents = node.borrow().parent.clone();
            for (i, parent) in parents.iter().enumerate() {
                let child_i = if parent.borrow().child(0).borrow().id == id { 0 } else { 1 };
                node.borrow_mut().in_prob[i] = hadamard(&parent.borrow().in_prob(), &parent.borrow().p(child_i));
            }
        }
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
    let input = Rc::new(input);
    // Solver 初期化・実行
    let mut solver = Solver::new(&input, 0, 1800);
    solver = solver.solve();
    solver.ans();
    solver.result();
}
