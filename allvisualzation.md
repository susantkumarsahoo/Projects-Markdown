# Visualization gallery

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualization Gallery</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.18.0.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: white;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            text-align: center;
            color: rgba(255,255,255,0.9);
            margin-bottom: 40px;
            font-size: 1.1em;
        }
        .section {
            margin-bottom: 50px;
        }
        .section-title {
            color: white;
            font-size: 2em;
            margin-bottom: 25px;
            padding-bottom: 10px;
            border-bottom: 3px solid rgba(255,255,255,0.3);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 25px;
        }
        .chart-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .chart-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        }
        .chart-title {
            font-size: 1.4em;
            margin-bottom: 8px;
            color: #667eea;
            font-weight: 600;
        }
        .chart-desc {
            color: #666;
            margin-bottom: 20px;
            font-size: 0.95em;
            line-height: 1.5;
        }
        .chart-container {
            position: relative;
            height: 350px;
        }
        canvas {
            max-height: 100%;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Interactive Visualization Gallery</h1>
        <p class="subtitle">Explore common 2D, hierarchical, flow, and 3D visualizations with live examples</p>

          <!-- 2D CHARTS -->
        <div class="section">
            <h2 class="section-title">📈 2D Charts</h2>
            <div class="grid">
                <div class="chart-card">
                    <h3 class="chart-title">Pie Chart</h3>
                    <p class="chart-desc">A basic composition chart showing category shares</p>
                    <div class="chart-container">
                        <canvas id="pieChart"></canvas>
                    </div>
                </div>

                <div class="chart-card">
                    <h3 class="chart-title">Donut Chart</h3>
                    <p class="chart-desc">A pie with a center hole—great for adding a total or key metric</p>
                    <div class="chart-container">
                        <canvas id="donutChart"></canvas>
                    </div>
                </div>

                <div class="chart-card">
                    <h3 class="chart-title">Polar Area Chart</h3>
                    <p class="chart-desc">Radial bars sized by value—seasonality or cyclical data</p>
                    <div class="chart-container">
                        <canvas id="polarChart"></canvas>
                    </div>
                </div>

                <div class="chart-card">
                    <h3 class="chart-title">Multi-line Chart</h3>
                    <p class="chart-desc">Multiple series over time—comparative trends</p>
                    <div class="chart-container">
                        <canvas id="multilineChart"></canvas>
                    </div>
                </div>

                <div class="chart-card">
                    <h3 class="chart-title">Stacked Area Chart</h3>
                    <p class="chart-desc">Areas stacked to show parts of a whole over time</p>
                    <div class="chart-container">
                        <canvas id="stackedAreaChart"></canvas>
                    </div>
                </div>

                <div class="chart-card">
                    <h3 class="chart-title">Scatter Plot</h3>
                    <p class="chart-desc">Points colored by category—clusters and relationships</p>
                    <div class="chart-container">
                        <canvas id="scatterChart"></canvas>
                    </div>
                </div>

                <div class="chart-card">
                    <h3 class="chart-title">Bar Chart</h3>
                    <p class="chart-desc">Horizontal bars for category comparison</p>
                    <div class="chart-container">
                        <canvas id="barChart"></canvas>
                    </div>
                </div>

                <div class="chart-card">
                    <h3 class="chart-title">Stacked Bar Chart</h3>
                    <p class="chart-desc">Bars stacked to show composition across categories</p>
                    <div class="chart-container">
                        <canvas id="stackedBarChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <!-- HIERARCHICAL & FLOW -->
        <div class="section">
            <h2 class="section-title">🌳 Hierarchical & Flow Charts</h2>
            <div class="grid">
                <div class="chart-card">
                    <h3 class="chart-title">Sunburst Chart</h3>
                    <p class="chart-desc">Radial hierarchy—nested categories and proportions</p>
                    <div class="chart-container" id="sunburstChart"></div>
                </div>

                <div class="chart-card">
                    <h3 class="chart-title">Sankey Diagram</h3>
                    <p class="chart-desc">Flows between nodes—volume shown by link thickness</p>
                    <div class="chart-container" id="sankeyChart"></div>
                </div>

                <div class="chart-card">
                    <h3 class="chart-title">Treemap</h3>
                    <p class="chart-desc">Rectangles sized by value—hierarchical proportions</p>
                    <div class="chart-container" id="treemapChart"></div>
                </div>

                <div class="chart-card">
                    <h3 class="chart-title">Network Diagram</h3>
                    <p class="chart-desc">Nodes and edges—relationships and connections</p>
                    <div class="chart-container">
                        <canvas id="networkChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <!-- 3D CHARTS -->
        <div class="section">
            <h2 class="section-title">🎲 3D Charts</h2>
            <div class="grid">
                <div class="chart-card">
                    <h3 class="chart-title">3D Scatter Plot</h3>
                    <p class="chart-desc">Points in 3D space—clusters and planes</p>
                    <div class="chart-container" id="scatter3dChart"></div>
                </div>

                <div class="chart-card">
                    <h3 class="chart-title">3D Surface Plot</h3>
                    <p class="chart-desc">Continuous surface in 3D—showing mathematical functions</p>
                    <div class="chart-container" id="surface3dChart"></div>
                </div>

                <div class="chart-card">
                    <h3 class="chart-title">3D Bar Chart</h3>
                    <p class="chart-desc">Bars extruded in 3D—category comparisons with depth</p>
                    <div class="chart-container" id="bar3dChart"></div>
                </div>

                <div class="chart-card">
                    <h3 class="chart-title">3D Bubble Chart</h3>
                    <p class="chart-desc">Sized spheres—value encoded by radius in 3D</p>
                    <div class="chart-container" id="bubble3dChart"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Color palettes
        const colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a', '#fee140', '#30cfd0'];
        const colors2 = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'];

        // Pie Chart
        new Chart(document.getElementById('pieChart'), {
            type: 'pie',
            data: {
                labels: ['Marketing', 'Sales', 'Engineering', 'Support', 'Operations'],
                datasets: [{
                    data: [30, 25, 20, 15, 10],
                    backgroundColor: colors.slice(0, 5)
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });

        // Donut Chart
        new Chart(document.getElementById('donutChart'), {
            type: 'doughnut',
            data: {
                labels: ['Desktop', 'Mobile', 'Tablet', 'Other'],
                datasets: [{
                    data: [45, 35, 15, 5],
                    backgroundColor: colors2.slice(0, 4)
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });

        // Polar Area Chart
        new Chart(document.getElementById('polarChart'), {
            type: 'polarArea',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [{
                    data: [11, 16, 7, 14, 20, 18],
                    backgroundColor: colors.slice(0, 6).map(c => c + '80')
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });

        // Multi-line Chart
        new Chart(document.getElementById('multilineChart'), {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
                datasets: [{
                    label: 'Product A',
                    data: [65, 59, 80, 81, 56, 55, 70],
                    borderColor: colors[0],
                    backgroundColor: colors[0] + '20',
                    tension: 0.4
                }, {
                    label: 'Product B',
                    data: [28, 48, 40, 19, 86, 27, 90],
                    borderColor: colors[1],
                    backgroundColor: colors[1] + '20',
                    tension: 0.4
                }, {
                    label: 'Product C',
                    data: [45, 25, 60, 55, 65, 75, 80],
                    borderColor: colors[2],
                    backgroundColor: colors[2] + '20',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });

        // Stacked Area Chart
        new Chart(document.getElementById('stackedAreaChart'), {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [{
                    label: 'Email',
                    data: [30, 35, 33, 38, 42, 45],
                    backgroundColor: colors[0] + '80',
                    borderColor: colors[0],
                    fill: true
                }, {
                    label: 'Social',
                    data: [20, 25, 28, 30, 33, 35],
                    backgroundColor: colors[1] + '80',
                    borderColor: colors[1],
                    fill: true
                }, {
                    label: 'Direct',
                    data: [15, 18, 20, 22, 25, 28],
                    backgroundColor: colors[2] + '80',
                    borderColor: colors[2],
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom' }
                },
                scales: {
                    y: { stacked: true }
                }
            }
        });

        // Scatter Chart
        new Chart(document.getElementById('scatterChart'), {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Group A',
                    data: Array.from({length: 20}, () => ({x: Math.random()*100, y: Math.random()*100})),
                    backgroundColor: colors[0]
                }, {
                    label: 'Group B',
                    data: Array.from({length: 20}, () => ({x: Math.random()*100, y: Math.random()*100})),
                    backgroundColor: colors[3]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });

        // Bar Chart
        new Chart(document.getElementById('barChart'), {
            type: 'bar',
            data: {
                labels: ['Q1', 'Q2', 'Q3', 'Q4'],
                datasets: [{
                    label: 'Revenue ($M)',
                    data: [120, 150, 180, 200],
                    backgroundColor: colors.slice(0, 4)
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });

        // Stacked Bar Chart
        new Chart(document.getElementById('stackedBarChart'), {
            type: 'bar',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
                datasets: [{
                    label: 'Region A',
                    data: [12, 19, 15, 17, 20],
                    backgroundColor: colors[0]
                }, {
                    label: 'Region B',
                    data: [8, 11, 13, 10, 15],
                    backgroundColor: colors[1]
                }, {
                    label: 'Region C',
                    data: [5, 8, 10, 12, 11],
                    backgroundColor: colors[2]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom' }
                },
                scales: {
                    x: { stacked: true },
                    y: { stacked: true }
                }
            }
        });

        // Network Chart (simple simulation)
        const networkCtx = document.getElementById('networkChart').getContext('2d');
        const networkCanvas = document.getElementById('networkChart');
        networkCanvas.width = networkCanvas.offsetWidth;
        networkCanvas.height = networkCanvas.offsetHeight;
        
        const nodes = Array.from({length: 15}, (_, i) => ({
            x: Math.random() * networkCanvas.width,
            y: Math.random() * networkCanvas.height,
            r: 8 + Math.random() * 12,
            color: colors[i % colors.length]
        }));

        function drawNetwork() {
            networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);
            
            // Draw edges
            networkCtx.strokeStyle = '#ddd';
            networkCtx.lineWidth = 1;
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const dist = Math.hypot(nodes[i].x - nodes[j].x, nodes[i].y - nodes[j].y);
                    if (dist < 150) {
                        networkCtx.beginPath();
                        networkCtx.moveTo(nodes[i].x, nodes[i].y);
                        networkCtx.lineTo(nodes[j].x, nodes[j].y);
                        networkCtx.stroke();
                    }
                }
            }
            
            // Draw nodes
            nodes.forEach(node => {
                networkCtx.beginPath();
                networkCtx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
                networkCtx.fillStyle = node.color;
                networkCtx.fill();
                networkCtx.strokeStyle = '#fff';
                networkCtx.lineWidth = 2;
                networkCtx.stroke();
            });
        }
        drawNetwork();

        // Plotly charts
        // Sunburst
        Plotly.newPlot('sunburstChart', [{
            type: 'sunburst',
            labels: ['Total', 'A', 'B', 'C', 'A1', 'A2', 'B1', 'B2', 'C1'],
            parents: ['', 'Total', 'Total', 'Total', 'A', 'A', 'B', 'B', 'C'],
            values: [100, 40, 35, 25, 20, 20, 18, 17, 25],
            marker: { colors: colors }
        }], {
            margin: { l: 0, r: 0, b: 0, t: 0 },
            height: 350
        }, {responsive: true});

        // Sankey
        Plotly.newPlot('sankeyChart', [{
            type: 'sankey',
            node: {
                label: ['Source A', 'Source B', 'Process 1', 'Process 2', 'Output'],
                color: colors,
                pad: 15
            },
            link: {
                source: [0, 0, 1, 1, 2, 3],
                target: [2, 3, 2, 3, 4, 4],
                value: [8, 4, 5, 7, 10, 8]
            }
        }], {
            margin: { l: 0, r: 0, b: 0, t: 0 },
            height: 350
        }, {responsive: true});

        // Treemap
        Plotly.newPlot('treemapChart', [{
            type: 'treemap',
            labels: ['Total', 'Marketing', 'Sales', 'Tech', 'Email', 'Social', 'Inside', 'Outside', 'Dev', 'QA'],
            parents: ['', 'Total', 'Total', 'Total', 'Marketing', 'Marketing', 'Sales', 'Sales', 'Tech', 'Tech'],
            values: [100, 30, 35, 35, 15, 15, 20, 15, 20, 15],
            marker: { colors: colors }
        }], {
            margin: { l: 0, r: 0, b: 0, t: 0 },
            height: 350
        }, {responsive: true});

        // 3D Scatter
        const scatter3dData = [{
            type: 'scatter3d',
            mode: 'markers',
            x: Array.from({length: 50}, () => Math.random() * 10),
            y: Array.from({length: 50}, () => Math.random() * 10),
            z: Array.from({length: 50}, () => Math.random() * 10),
            marker: {
                size: 5,
                color: Array.from({length: 50}, () => Math.random() * 10),
                colorscale: 'Viridis'
            }
        }];
        Plotly.newPlot('scatter3dChart', scatter3dData, {
            margin: { l: 0, r: 0, b: 0, t: 0 },
            height: 350
        }, {responsive: true});

        // 3D Surface
        const size = 25;
        const x = [], y = [], z = [];
        for (let i = 0; i < size; i++) {
            x.push(i);
            y.push(i);
        }
        for (let i = 0; i < size; i++) {
            z[i] = [];
            for (let j = 0; j < size; j++) {
                z[i][j] = Math.sin(i/3) * Math.cos(j/3) * 5;
            }
        }
        Plotly.newPlot('surface3dChart', [{
            type: 'surface',
            x: x,
            y: y,
            z: z,
            colorscale: 'Viridis'
        }], {
            margin: { l: 0, r: 0, b: 0, t: 0 },
            height: 350
        }, {responsive: true});

        // 3D Bar
        Plotly.newPlot('bar3dChart', [{
            type: 'bar3d',
            x: ['A', 'B', 'C', 'D'],
            y: ['X', 'Y', 'Z'],
            z: [
                [12, 18, 15, 20],
                [8, 14, 10, 16],
                [15, 11, 18, 13]
            ]
        }], {
            margin: { l: 0, r: 0, b: 0, t: 0 },
            height: 350
        }, {responsive: true});

        // 3D Bubble
        Plotly.newPlot('bubble3dChart', [{
            type: 'scatter3d',
            mode: 'markers',
            x: Array.from({length: 30}, () => Math.random() * 10),
            y: Array.from({length: 30}, () => Math.random() * 10),
            z: Array.from({length: 30}, () => Math.random() * 10),
            marker: {
                size: Array.from({length: 30}, () => 5 + Math.random() * 20),
                color: colors,
                opacity: 0.7
            }
        }], {
            margin: { l: 0, r: 0, b: 0, t: 0 },
            height: 350
        }, {responsive: true});
    </script>
</body>
</html>
