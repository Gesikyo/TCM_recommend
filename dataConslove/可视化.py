import json
import networkx as nx
import plotly.graph_objects as go

# 读取 JSON 文件
with open('CF_utf-8_modified.json', 'r', encoding='utf-8') as f:
    prescriptions_data = json.load(f)

# 创建一个空的有向图
graph = nx.DiGraph()

# 遍历每个方剂及其信息
for prescription, info in prescriptions_data.items():
    # 添加方剂节点
    graph.add_node(prescription, type='prescription', diseases=info['disease'])

    # 添加成分节点和边
    components = info['components']
    for component_dict in components:
        for component, quantity in component_dict.items():
            if not graph.has_node(component):
                graph.add_node(component, type='component')
            graph.add_edge(prescription, component, quantity=quantity)

# 使用 Plotly 创建交互式图谱
pos = nx.spring_layout(graph, k=1.5)  # 定义布局，调整 k 值以影响节点之间的距离

# 节点位置和标签
node_x = []
node_y = []
node_text = []
for node, (x, y) in pos.items():
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)

# 创建节点的 Scatter trace
node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    text=node_text,
    hoverinfo='text',
    marker=dict(
        showscale=False,
        colorscale='Blues',
        size=10,
        line_width=2
    )
)

# 边的位置和形状
edge_x = []
edge_y = []
for edge in graph.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

# 创建边的 Scatter trace
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1, color='#888'),
    hoverinfo='none',
    mode='lines'
)

# 创建图形布局
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Prescriptions Knowledge Graph',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                ))

# 显示图形
fig.show()
