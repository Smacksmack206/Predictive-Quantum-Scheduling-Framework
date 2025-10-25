// Battery History Visualization
class BatteryHistoryDashboard {
    constructor() {
        this.chart = null;
        this.currentRange = 'today';
        this.showEAS = true;
        this.showApps = true;
        this.updateInterval = null;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.setupChart();
        this.loadData();
        this.startLiveUpdates();
        this.checkForUpdates();
    }
    
    setupEventListeners() {
        // Theme selector
        document.getElementById('themeSelect').addEventListener('change', (e) => {
            this.changeTheme(e.target.value);
        });
        
        // Time range buttons
        document.querySelectorAll('.range-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.changeTimeRange(e.target.dataset.range);
            });
        });
        
        // Toggle switches
        document.getElementById('showEAS').addEventListener('change', (e) => {
            this.showEAS = e.target.checked;
            this.updateChart();
        });
        
        document.getElementById('showApps').addEventListener('change', (e) => {
            this.showApps = e.target.checked;
            this.updateAppChanges();
        });
        
        // Update modal
        document.getElementById('closeUpdateModal').addEventListener('click', () => {
            this.hideUpdateModal();
        });
        
        document.getElementById('skipUpdate').addEventListener('click', () => {
            this.skipUpdate();
        });
        
        document.getElementById('installUpdate').addEventListener('click', () => {
            this.installUpdate();
        });
    }
    
    changeTheme(theme) {
        document.body.className = `theme-${theme}`;
        localStorage.setItem('battery-optimizer-theme', theme);
        
        // Update chart colors
        if (this.chart) {
            this.updateChartTheme();
        }
    }
    
    changeTimeRange(range) {
        this.currentRange = range;
        
        // Update active button
        document.querySelectorAll('.range-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-range="${range}"]`).classList.add('active');
        
        this.loadData();
    }
    
    setupChart() {
        const ctx = document.getElementById('batteryChart').getContext('2d');
        
        console.log('Setting up chart...');
        
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Battery Level (%)',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4,
                        yAxisID: 'y',
                        pointRadius: 1,
                        pointHoverRadius: 4
                    },
                    {
                        label: 'Drain Rate (mA)',
                        data: [],
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
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
                        display: false // We have custom legend
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#374151',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: true,
                        callbacks: {
                            title: (context) => {
                                return new Date(context[0].parsed.x).toLocaleString();
                            },
                            label: (context) => {
                                const label = context.dataset.label;
                                const value = context.parsed.y;
                                
                                if (label === 'Battery Level (%)') {
                                    return `${label}: ${value.toFixed(1)}%`;
                                } else if (label === 'Drain Rate (mA)') {
                                    return `${label}: ${value.toFixed(0)}mA`;
                                } else if (label === 'EAS Status') {
                                    return `EAS: ${value > 0.5 ? 'Active' : 'Inactive'}`;
                                }
                                return `${label}: ${value}`;
                            }
                        }
                    },
                    zoom: {
                        zoom: {
                            wheel: {
                                enabled: true,
                            },
                            pinch: {
                                enabled: true
                            },
                            mode: 'x',
                        },
                        pan: {
                            enabled: true,
                            mode: 'x',
                        }
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
    }
    
    updateChartTheme() {
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
            
            this.updateChart(data.history || []);
            this.updateCycles(data.cycles || []);
            this.updateAppChanges(data.app_changes || []);
            this.updateStatistics(data.statistics || {});
            
        } catch (error) {
            console.error('Failed to load battery history:', error);
            // Show error in UI
            document.getElementById('avgBatteryLife').textContent = 'Error';
            document.getElementById('avgDrainRate').textContent = 'Error';
            document.getElementById('easUptime').textContent = 'Error';
            document.getElementById('totalSavings').textContent = 'Error';
        }
    }
    
    updateChart(historyData = []) {
        if (!this.chart) {
            console.error('Chart not initialized');
            return;
        }
        
        console.log(`Updating chart with ${historyData.length} data points`);
        
        // Process data for chart
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
        });
        
        console.log(`Processed data - Battery: ${batteryData.length}, Drain: ${drainData.length}, EAS: ${easData.length}`);
        
        // Update chart data
        this.chart.data.datasets[0].data = batteryData;
        this.chart.data.datasets[1].data = drainData;
        this.chart.data.datasets[2].data = this.showEAS ? easData : [];
        
        // Force chart update
        this.chart.update('active');
        
        console.log('Chart updated successfully');
    }
    
    updateCycles(cyclesData = []) {
        const cyclesGrid = document.getElementById('cyclesGrid');
        cyclesGrid.innerHTML = '';
        
        cyclesData.forEach(cycle => {
            const cycleCard = document.createElement('div');
            cycleCard.className = 'cycle-card';
            
            const startDate = new Date(cycle.start_time);
            const endDate = new Date(cycle.end_time);
            const duration = Math.round((endDate - startDate) / (1000 * 60 * 60 * 100)) / 10; // Hours with 1 decimal
            
            cycleCard.innerHTML = `
                <div class="cycle-header">
                    <div class="cycle-date">${startDate.toLocaleDateString()}</div>
                    <div class="cycle-duration">${duration}h</div>
                </div>
                <div class="cycle-stats">
                    <div class="cycle-stat">
                        <div class="cycle-stat-value">${cycle.start_level}%</div>
                        <div class="cycle-stat-label">Start</div>
                    </div>
                    <div class="cycle-stat">
                        <div class="cycle-stat-value">${cycle.end_level}%</div>
                        <div class="cycle-stat-label">End</div>
                    </div>
                    <div class="cycle-stat">
                        <div class="cycle-stat-value">${cycle.avg_drain_rate}mA</div>
                        <div class="cycle-stat-label">Avg Drain</div>
                    </div>
                    <div class="cycle-stat">
                        <div class="cycle-stat-value">${cycle.eas_uptime}%</div>
                        <div class="cycle-stat-label">EAS Uptime</div>
                    </div>
                </div>
                <div class="cycle-eas-status">
                    <div class="eas-indicator ${cycle.eas_uptime > 50 ? 'active' : 'inactive'}"></div>
                    <span>EAS ${cycle.eas_uptime > 50 ? 'Mostly Active' : 'Mostly Inactive'}</span>
                </div>
            `;
            
            cyclesGrid.appendChild(cycleCard);
        });
    }
    
    updateAppChanges(appChangesData = []) {
        if (!this.showApps) {
            document.getElementById('appChangesTimeline').style.display = 'none';
            return;
        }
        
        document.getElementById('appChangesTimeline').style.display = 'block';
        const timeline = document.getElementById('appChangesTimeline');
        timeline.innerHTML = '';
        
        appChangesData.forEach(change => {
            const timelineItem = document.createElement('div');
            timelineItem.className = 'timeline-item';
            
            const changeDate = new Date(change.timestamp);
            
            timelineItem.innerHTML = `
                <div class="timeline-marker">
                    <span class="material-icons">apps</span>
                </div>
                <div class="timeline-content">
                    <div class="timeline-header">
                        <div class="timeline-title">${change.change_type}</div>
                        <div class="timeline-date">${changeDate.toLocaleString()}</div>
                    </div>
                    <div class="timeline-description">${change.description}</div>
                    <div class="app-list">
                        ${change.apps_added.map(app => `<span class="app-tag added">+${app}</span>`).join('')}
                        ${change.apps_removed.map(app => `<span class="app-tag removed">-${app}</span>`).join('')}
                        ${change.apps_unchanged.map(app => `<span class="app-tag">${app}</span>`).join('')}
                    </div>
                </div>
            `;
            
            timeline.appendChild(timelineItem);
        });
    }
    
    updateStatistics(stats = {}) {
        console.log('Updating statistics:', stats);
        
        const avgBatteryLife = stats.avg_battery_life || 0;
        const avgDrainRate = stats.avg_drain_rate || 0;
        const easUptime = stats.eas_uptime || 0;
        const totalSavings = stats.total_savings || 0;
        
        document.getElementById('avgBatteryLife').textContent = `${avgBatteryLife}h`;
        document.getElementById('avgDrainRate').textContent = `${Math.round(avgDrainRate)}mA`;
        document.getElementById('easUptime').textContent = `${Math.round(easUptime)}%`;
        document.getElementById('totalSavings').textContent = `${totalSavings}h`;
        
        console.log(`Statistics updated - Battery Life: ${avgBatteryLife}h, Drain Rate: ${avgDrainRate}mA, EAS Uptime: ${easUptime}%, Savings: ${totalSavings}h`);
    }
    
    startLiveUpdates() {
        // Update every 30 seconds
        this.updateInterval = setInterval(() => {
            this.loadData();
        }, 30000);
    }
    
    async checkForUpdates() {
        try {
            const response = await fetch('/api/check-updates');
            const updateInfo = await response.json();
            
            const updateStatus = document.getElementById('updateStatus');
            const updateText = document.getElementById('updateText');
            
            if (updateInfo.update_available) {
                updateText.textContent = `Update available: v${updateInfo.latest_version}`;
                updateStatus.style.color = 'var(--accent-warning)';
                updateStatus.style.cursor = 'pointer';
                updateStatus.addEventListener('click', () => {
                    this.showUpdateModal(updateInfo);
                });
            } else {
                updateText.textContent = 'Up to date';
                updateStatus.style.color = 'var(--accent-success)';
            }
        } catch (error) {
            console.error('Failed to check for updates:', error);
            document.getElementById('updateText').textContent = 'Update check failed';
        }
    }
    
    showUpdateModal(updateInfo) {
        const modal = document.getElementById('updateModal');
        const updateInfoDiv = document.getElementById('updateInfo');
        
        updateInfoDiv.innerHTML = `
            <h4>Version ${updateInfo.latest_version}</h4>
            <p><strong>Current:</strong> v${updateInfo.current_version}</p>
            <p><strong>Release Date:</strong> ${new Date(updateInfo.release_date).toLocaleDateString()}</p>
            <div style="margin-top: 16px;">
                <h5>What's New:</h5>
                <ul>
                    ${updateInfo.changelog.map(item => `<li>${item}</li>`).join('')}
                </ul>
            </div>
            <div style="margin-top: 16px;">
                <p><strong>Download Size:</strong> ${updateInfo.download_size}</p>
                <p><strong>Type:</strong> ${updateInfo.update_type}</p>
            </div>
        `;
        
        modal.classList.add('show');
    }
    
    hideUpdateModal() {
        document.getElementById('updateModal').classList.remove('show');
    }
    
    async skipUpdate() {
        try {
            await fetch('/api/skip-update', { method: 'POST' });
            this.hideUpdateModal();
        } catch (error) {
            console.error('Failed to skip update:', error);
        }
    }
    
    async installUpdate() {
        try {
            const installBtn = document.getElementById('installUpdate');
            const originalText = installBtn.innerHTML;
            
            installBtn.innerHTML = '<span class="material-icons">download</span>Installing...';
            installBtn.disabled = true;
            
            const response = await fetch('/api/install-update', { method: 'POST' });
            const result = await response.json();
            
            if (result.success) {
                installBtn.innerHTML = '<span class="material-icons">check</span>Update Installed';
                setTimeout(() => {
                    // App will restart automatically
                    window.location.reload();
                }, 2000);
            } else {
                installBtn.innerHTML = originalText;
                installBtn.disabled = false;
                alert('Update failed: ' + result.error);
            }
        } catch (error) {
            console.error('Failed to install update:', error);
            document.getElementById('installUpdate').disabled = false;
        }
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('Battery History Dashboard initializing...');
    
    // Load saved theme
    const savedTheme = localStorage.getItem('battery-optimizer-theme') || 'dark';
    document.body.className = `theme-${savedTheme}`;
    
    const themeSelect = document.getElementById('themeSelect');
    if (themeSelect) {
        themeSelect.value = savedTheme;
    }
    
    console.log(`Theme loaded: ${savedTheme}`);
    
    // Initialize dashboard
    new BatteryHistoryDashboard();
});