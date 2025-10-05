// Simple Battery History Chart - No Syntax Errors
class BatteryHistoryDashboard {
    constructor() {
        this.chart = null;
        this.currentRange = 'today';
        this.init();
    }
    
    init() {
        console.log('Initializing Battery History Dashboard...');
        this.setupEventListeners();
        this.setupChart();
        this.loadData();
        this.startUpdates();
    }
    
    setupEventListeners() {
        // Theme selector
        const themeSelect = document.getElementById('themeSelect');
        if (themeSelect) {
            themeSelect.addEventListener('change', (e) => {
                this.changeTheme(e.target.value);
            });
        }
        
        // Time range buttons
        document.querySelectorAll('.range-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.changeTimeRange(e.target.dataset.range);
            });
        });
    }
    
    changeTheme(theme) {
        document.body.className = 'theme-' + theme;
        localStorage.setItem('battery-optimizer-theme', theme);
        if (this.chart) {
            this.chart.update();
        }
    }
    
    changeTimeRange(range) {
        this.currentRange = range;
        document.querySelectorAll('.range-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector('[data-range="' + range + '"]').classList.add('active');
        this.loadData();
    }
    
    setupChart() {
        const canvas = document.getElementById('batteryChart');
        if (!canvas) {
            console.error('Chart canvas not found');
            return;
        }
        
        const ctx = canvas.getContext('2d');
        
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Battery Level (%)',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    fill: true
                }, {
                    label: 'Current Draw (mA)',
                    data: [],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            displayFormats: {
                                hour: 'HH:mm',
                                day: 'MMM dd'
                            }
                        }
                    },
                    y: {
                        type: 'linear',
                        position: 'left',
                        min: 0,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Battery Level (%)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        position: 'right',
                        min: 0,
                        grid: {
                            drawOnChartArea: false
                        },
                        title: {
                            display: true,
                            text: 'Current Draw (mA)'
                        }
                    }
                }
            }
        });
        
        console.log('Chart setup complete');
    }
    
    async loadData() {
        try {
            console.log('Loading data for range:', this.currentRange);
            const response = await fetch('/api/battery-history?range=' + this.currentRange);
            const data = await response.json();
            
            console.log('Loaded', data.history ? data.history.length : 0, 'data points');
            
            this.updateChart(data.history || []);
            this.updateStatistics(data.statistics || {});
            
        } catch (error) {
            console.error('Failed to load data:', error);
        }
    }
    
    updateChart(historyData) {
        if (!this.chart || !historyData) return;
        
        const batteryData = [];
        const drainData = [];
        
        historyData.forEach(point => {
            const timestamp = new Date(point.timestamp);
            
            batteryData.push({
                x: timestamp,
                y: point.battery_level
            });
            
            if (point.current_draw > 0) {
                drainData.push({
                    x: timestamp,
                    y: point.current_draw
                });
            }
        });
        
        this.chart.data.datasets[0].data = batteryData;
        this.chart.data.datasets[1].data = drainData;
        this.chart.update();
        
        console.log('Chart updated with', batteryData.length, 'battery points and', drainData.length, 'drain points');
    }
    
    updateStatistics(stats) {
        const elements = {
            'avgBatteryLife': (stats.avg_battery_life || 0) + 'h',
            'avgDrainRate': Math.round(stats.avg_drain_rate || 0) + 'mA',
            'easUptime': Math.round(stats.eas_uptime || 0) + '%',
            'totalSavings': (stats.total_savings || 0) + 'h'
        };
        
        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });
        
        console.log('Statistics updated:', stats);
    }
    
    startUpdates() {
        setInterval(() => {
            this.loadData();
        }, 5000);
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Load saved theme
    const savedTheme = localStorage.getItem('battery-optimizer-theme') || 'dark';
    document.body.className = 'theme-' + savedTheme;
    
    const themeSelect = document.getElementById('themeSelect');
    if (themeSelect) {
        themeSelect.value = savedTheme;
    }
    
    // Initialize dashboard
    new BatteryHistoryDashboard();
});