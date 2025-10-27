// Simplified Battery History Dashboard
class BatteryHistoryDashboard {
    constructor() {
        this.chart = null;
        this.currentRange = 'today';
        this.showEAS = true;
        this.showApps = true;
        this.updateInterval = null;
        
        console.log('Initializing Battery History Dashboard...');
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.setupChart();
        this.loadData();
        this.startLiveUpdates();
    }
    
    setupEventListeners() {
        // Theme selector
        const themeSelect = document.getElementById('themeSelect');
        if (themeSelect) {
            console.log('Theme selector found, adding event listener');
            themeSelect.addEventListener('change', (e) => {
                console.log(`Theme selector changed to: ${e.target.value}`);
                this.changeTheme(e.target.value);
                // Force immediate visual update
                setTimeout(() => {
                    if (this.chart) {
                        this.updateChartTheme();
                    }
                }, 100);
            });
        } else {
            console.error('Theme selector not found!');
        }
        
        // Time range buttons
        document.querySelectorAll('.range-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.changeTimeRange(e.target.dataset.range);
            });
        });
        
        // Toggle switches
        const showEASToggle = document.getElementById('showEAS');
        if (showEASToggle) {
            showEASToggle.addEventListener('change', (e) => {
                this.showEAS = e.target.checked;
                this.updateChart();
            });
        }
        
        const showAppsToggle = document.getElementById('showApps');
        if (showAppsToggle) {
            showAppsToggle.addEventListener('change', (e) => {
                this.showApps = e.target.checked;
                this.updateAppChanges();
            });
        }
    }
    
    changeTheme(theme) {
        console.log(`Changing theme to: ${theme}`);
        
        // Force immediate theme change
        document.body.className = `theme-${theme}`;
        document.documentElement.className = `theme-${theme}`;
        
        // Save theme
        localStorage.setItem('battery-optimizer-theme', theme);
        
        // Force CSS re-evaluation
        document.body.style.display = 'none';
        document.body.offsetHeight; // Trigger reflow
        document.body.style.display = '';
        
        // Update chart colors if chart exists
        if (this.chart) {
            console.log('Updating chart theme...');
            setTimeout(() => {
                this.updateChartTheme();
                this.chart.update('none');
            }, 50);
        }
        
        // Force theme selector to show correct value
        const themeSelect = document.getElementById('themeSelect');
        if (themeSelect) {
            themeSelect.value = theme;
        }
        
        console.log(`Theme changed successfully to: ${theme}`);
    }
    
    changeTimeRange(range) {
        this.currentRange = range;
        
        // Update active button
        document.querySelectorAll('.range-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        const activeBtn = document.querySelector(`[data-range="${range}"]`);
        if (activeBtn) {
            activeBtn.classList.add('active');
        }
        
        console.log(`Changed time range to: ${range}`);
        this.loadData();
    }
    
    setupChart() {
        const canvas = document.getElementById('batteryChart');
        if (!canvas) {
            console.error('Chart canvas not found!');
            return;
        }
        
        const ctx = canvas.getContext('2d');
        console.log('Setting up chart...');
        
        // Ensure Chart.js is loaded
        if (typeof Chart === 'undefined') {
            console.error('Chart.js not loaded!');
            return;
        }
        
        try {
            this.chart = new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: 'Battery Level (%)',
                            data: [],
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.3,
                            yAxisID: 'y',
                            pointRadius: 1,
                            pointHoverRadius: 4
                        },
                        {
                            label: 'Current Draw (mA)',
                            data: [],
                            borderColor: '#f59e0b',
                            backgroundColor: 'rgba(245, 158, 11, 0.1)',
                            borderWidth: 2,
                            fill: false,
                            tension: 0.3,
                            yAxisID: 'y1',
                            pointRadius: 1,
                            pointHoverRadius: 4
                        },
                        {
                            label: 'EAS Status',
                            data: [],
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.2)',
                            borderWidth: 0,
                            fill: true,
                            stepped: true,
                            yAxisID: 'y2',
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleColor: '#ffffff',
                            bodyColor: '#ffffff',
                            borderColor: '#374151',
                            borderWidth: 1,
                            cornerRadius: 8,
                            displayColors: true
                        }
                    },
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                displayFormats: {
                                    hour: 'HH:mm',
                                    day: 'MMM dd',
                                    week: 'MMM dd',
                                    month: 'MMM yyyy'
                                }
                            },
                            grid: {
                                color: 'rgba(156, 163, 175, 0.1)'
                            },
                            ticks: {
                                color: '#9ca3af'
                            }
                        },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            min: 0,
                            max: 100,
                            grid: {
                                color: 'rgba(156, 163, 175, 0.1)'
                            },
                            ticks: {
                                color: '#9ca3af',
                                callback: function(value) {
                                    return value + '%';
                                }
                            },
                            title: {
                                display: true,
                                text: 'Battery Level (%)',
                                color: '#9ca3af'
                            }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            min: 0,
                            grid: {
                                drawOnChartArea: false,
                            },
                            ticks: {
                                color: '#9ca3af',
                                callback: function(value) {
                                    return value + 'mA';
                                }
                            },
                            title: {
                                display: true,
                                text: 'Current Draw (mA)',
                                color: '#9ca3af'
                            }
                        },
                        y2: {
                            type: 'linear',
                            display: false,
                            min: 0,
                            max: 1
                        }
                    }
                }
            });
            
            console.log('Chart setup complete');
        } catch (error) {
            console.error('Chart setup failed:', error);
        }
    }
    
    updateChartTheme() {
        if (!this.chart) return;
        
        const theme = document.body.className.replace('theme-', '');
        const isDark = theme === 'dark' || theme === 'solarized';
        
        const gridColor = isDark ? 'rgba(75, 85, 99, 0.3)' : 'rgba(156, 163, 175, 0.2)';
        const textColor = isDark ? '#9ca3af' : '#6b7280';
        
        this.chart.options.scales.x.grid.color = gridColor;
        this.chart.options.scales.y.grid.color = gridColor;
        this.chart.options.scales.x.ticks.color = textColor;
        this.chart.options.scales.y.ticks.color = textColor;
        this.chart.options.scales.y1.ticks.color = textColor;
        this.chart.options.scales.y.title.color = textColor;
        this.chart.options.scales.y1.title.color = textColor;
        
        this.chart.update();
    }
    
    async loadData() {
        try {
            console.log(`Loading battery history data for range: ${this.currentRange}`);
            const response = await fetch(`/api/battery-history?range=${this.currentRange}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log(`Loaded ${data.history?.length || 0} history points`);
            console.log('Sample data:', data.history?.slice(0, 2));
            
            // Force chart update even with empty data
            this.updateChart(data.history || []);
            this.updateCycles(data.cycles || []);
            this.updateAppChanges(data.app_changes || []);
            this.updateStatistics(data.statistics || {});
            
            // Show success message
            if (data.history && data.history.length > 0) {
                console.log(`✅ Successfully loaded ${data.history.length} data points`);
            } else {
                console.warn('⚠️ No history data available');
            }
            
        } catch (error) {
            console.error('Failed to load battery history:', error);
            this.showError('Failed to load data');
        }
    }
    
    updateChart(historyData = []) {
        if (!this.chart) {
            console.error('Chart not initialized');
            return;
        }
        
        console.log(`Updating chart with ${historyData.length} data points`);
        
        const batteryData = [];
        const drainData = [];
        const easData = [];
        
        historyData.forEach((point, index) => {
            try {
                const timestamp = new Date(point.timestamp);
                
                if (isNaN(timestamp.getTime())) {
                    console.warn(`Invalid timestamp at index ${index}:`, point.timestamp);
                    return;
                }
                
                // Battery level data
                if (typeof point.battery_level === 'number') {
                    batteryData.push({
                        x: timestamp,
                        y: point.battery_level
                    });
                }
                
                // Current draw data
                if (typeof point.current_draw === 'number' && point.current_draw > 0) {
                    drainData.push({
                        x: timestamp,
                        y: point.current_draw
                    });
                }
                
                // EAS status data
                if (this.showEAS) {
                    easData.push({
                        x: timestamp,
                        y: point.eas_active ? 1 : 0
                    });
                }
            } catch (error) {
                console.warn(`Error processing data point ${index}:`, error, point);
            }
        });\n        console.log(`Processed data - Battery: ${batteryData.length}, Drain: ${drainData.length}, EAS: ${easData.length}`);\n        // Update chart data\n        this.chart.data.datasets[0].data = batteryData;\n        this.chart.data.datasets[1].data = drainData;\n        this.chart.data.datasets[2].data = this.showEAS ? easData : [];\n        \n        // Force chart update\n        this.chart.update('active');\n        \n        console.log('Chart updated successfully');\n    }\n    \n    updateCycles(cyclesData = []) {\n        const cyclesGrid = document.getElementById('cyclesGrid');\n        if (!cyclesGrid) return;\n        \n        cyclesGrid.innerHTML = '';\n        \n        if (cyclesData.length === 0) {\n            cyclesGrid.innerHTML = '<div class=\"no-data\">No battery cycles recorded yet</div>';\n            return;\n        }\n        \n        cyclesData.forEach(cycle => {\n            const cycleCard = document.createElement('div');\n            cycleCard.className = 'cycle-card';\n            \n            const startDate = new Date(cycle.start_time);\n            const endDate = new Date(cycle.end_time);\n            const duration = Math.round((endDate - startDate) / (1000 * 60 * 60 * 100)) / 10;\n            \n            cycleCard.innerHTML = `\n                <div class=\"cycle-header\">\n                    <div class=\"cycle-date\">${startDate.toLocaleDateString()}</div>\n                    <div class=\"cycle-duration\">${duration}h</div>\n                </div>\n                <div class=\"cycle-stats\">\n                    <div class=\"cycle-stat\">\n                        <div class=\"cycle-stat-value\">${cycle.start_level}%</div>\n                        <div class=\"cycle-stat-label\">Start</div>\n                    </div>\n                    <div class=\"cycle-stat\">\n                        <div class=\"cycle-stat-value\">${cycle.end_level}%</div>\n                        <div class=\"cycle-stat-label\">End</div>\n                    </div>\n                    <div class=\"cycle-stat\">\n                        <div class=\"cycle-stat-value\">${cycle.avg_drain_rate}mA</div>\n                        <div class=\"cycle-stat-label\">Avg Drain</div>\n                    </div>\n                    <div class=\"cycle-stat\">\n                        <div class=\"cycle-stat-value\">${cycle.eas_uptime}%</div>\n                        <div class=\"cycle-stat-label\">EAS Uptime</div>\n                    </div>\n                </div>\n                <div class=\"cycle-eas-status\">\n                    <div class=\"eas-indicator ${cycle.eas_uptime > 50 ? 'active' : 'inactive'}\"></div>\n                    <span>EAS ${cycle.eas_uptime > 50 ? 'Mostly Active' : 'Mostly Inactive'}</span>\n                </div>\n            `;\n            \n            cyclesGrid.appendChild(cycleCard);\n        });\n    }\n    \n    updateAppChanges(appChangesData = []) {\n        const timeline = document.getElementById('appChangesTimeline');\n        if (!timeline) return;\n        \n        if (!this.showApps) {\n            timeline.style.display = 'none';\n            return;\n        }\n        \n        timeline.style.display = 'block';\n        timeline.innerHTML = '';\n        \n        if (appChangesData.length === 0) {\n            timeline.innerHTML = '<div class=\"no-data\">No app configuration changes recorded</div>';\n            return;\n        }\n        \n        appChangesData.forEach(change => {\n            const timelineItem = document.createElement('div');\n            timelineItem.className = 'timeline-item';\n            \n            const changeDate = new Date(change.timestamp);\n            \n            timelineItem.innerHTML = `\n                <div class=\"timeline-marker\">\n                    <span class=\"material-icons\">apps</span>\n                </div>\n                <div class=\"timeline-content\">\n                    <div class=\"timeline-header\">\n                        <div class=\"timeline-title\">${change.change_type}</div>\n                        <div class=\"timeline-date\">${changeDate.toLocaleString()}</div>\n                    </div>\n                    <div class=\"timeline-description\">${change.description}</div>\n                    <div class=\"app-list\">\n                        ${change.apps_added?.map(app => `<span class=\"app-tag added\">+${app}</span>`).join('') || ''}\n                        ${change.apps_removed?.map(app => `<span class=\"app-tag removed\">-${app}</span>`).join('') || ''}\n                        ${change.apps_unchanged?.map(app => `<span class=\"app-tag\">${app}</span>`).join('') || ''}\n                    </div>\n                </div>\n            `;\n            \n            timeline.appendChild(timelineItem);\n        });\n    }\n    \n    updateStatistics(stats = {}) {\n        console.log('Updating statistics:', stats);\n        \n        const avgBatteryLife = stats.avg_battery_life || 0;\n        const avgDrainRate = stats.avg_drain_rate || 0;\n        const easUptime = stats.eas_uptime || 0;\n        const totalSavings = stats.total_savings || 0;\n        \n        const elements = {\n            'avgBatteryLife': `${avgBatteryLife}h`,\n            'avgDrainRate': `${Math.round(avgDrainRate)}mA`,\n            'easUptime': `${Math.round(easUptime)}%`,\n            'totalSavings': `${totalSavings}h`\n        };\n        \n        Object.entries(elements).forEach(([id, value]) => {\n            const element = document.getElementById(id);\n            if (element) {\n                element.textContent = value;\n            } else {\n                console.warn(`Element ${id} not found`);\n            }\n        });\n        \n        console.log(`Statistics updated - Battery Life: ${avgBatteryLife}h, Drain Rate: ${avgDrainRate}mA, EAS Uptime: ${easUptime}%, Savings: ${totalSavings}h`);\n    }\n    \n    showError(message) {\n        const errorElements = ['avgBatteryLife', 'avgDrainRate', 'easUptime', 'totalSavings'];\n        errorElements.forEach(id => {\n            const element = document.getElementById(id);\n            if (element) {\n                element.textContent = 'Error';\n            }\n        });\n        console.error('Dashboard error:', message);\n    }\n    \n    startLiveUpdates() {\n        // Update every 30 seconds\n        this.updateInterval = setInterval(() => {\n            console.log('Auto-refreshing data...');\n            this.loadData();\n        }, 30000);\n        \n        console.log('Live updates started (30s interval)');\n    }\n    \n    destroy() {\n        if (this.updateInterval) {\n            clearInterval(this.updateInterval);\n        }\n        if (this.chart) {\n            this.chart.destroy();\n        }\n    }\n}\n\n// Initialize dashboard when page loads\ndocument.addEventListener('DOMContentLoaded', () => {\n    console.log('Battery History Dashboard initializing...');\n    \n    // Load saved theme\n    const savedTheme = localStorage.getItem('battery-optimizer-theme') || 'dark';\n    document.body.className = `theme-${savedTheme}`;\n    \n    const themeSelect = document.getElementById('themeSelect');\n    if (themeSelect) {\n        themeSelect.value = savedTheme;\n    }\n    \n    console.log(`Theme loaded: ${savedTheme}`);\n    \n    // Initialize dashboard\n    window.batteryDashboard = new BatteryHistoryDashboard();\n});