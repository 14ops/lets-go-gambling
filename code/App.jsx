import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Separator } from '@/components/ui/separator.jsx'
import { 
  Activity, 
  BarChart3, 
  Brain, 
  Calculator, 
  ChartLine, 
  Cpu, 
  DollarSign, 
  Eye, 
  GamepadIcon, 
  LineChart, 
  PieChart, 
  Settings, 
  Shield, 
  Target, 
  TrendingUp, 
  Users, 
  Zap 
} from 'lucide-react'
import './App.css'

// Import visualization images
import strategyComparison from './assets/strategy_comparison.png'
import riskAnalysis from './assets/risk_analysis.png'
import bankrollProgressionTakeshi from './assets/bankroll_progression_takeshi.png'
import bankrollProgressionLelouch from './assets/bankroll_progression_lelouch.png'
import bankrollProgressionKazuya from './assets/bankroll_progression_kazuya.png'
import bankrollProgressionSenku from './assets/bankroll_progression_senku.png'
import characterCardTakeshi from './assets/character_card_takeshi.png'
import characterCardLelouch from './assets/character_card_lelouch.png'
import characterCardKazuya from './assets/character_card_kazuya.png'
import characterCardSenku from './assets/character_card_senku.png'
import strategyHeatmap from './assets/strategy_performance_heatmap.png'
import riskReturn3D from './assets/3d_risk_return_analysis.png'
import senkuDetailed from './assets/senku_detailed_analysis.png'
import parameterSensitivity from './assets/parameter_sensitivity_analysis.png'
import monteCarloResults from './assets/monte_carlo_simulation_results.png'
import probabilityTheory from './assets/probability_theory_visualization.png'
import gameMechanics from './assets/game_mechanics_analysis.png'
import detectionEvasion from './assets/detection_evasion_analysis.png'
import portfolioOptimization from './assets/portfolio_optimization_analysis.png'
import machineLearning from './assets/machine_learning_analysis.png'
import extremeScenarios from './assets/extreme_scenarios_analysis.png'
import behavioralPsychology from './assets/behavioral_psychology_analysis.png'
import technicalImplementation from './assets/technical_implementation_analysis.png'
import marketMicrostructure from './assets/market_microstructure_analysis.png'

function App() {
  const [activeStrategy, setActiveStrategy] = useState('hybrid')
  const [simulationRunning, setSimulationRunning] = useState(false)
  const [liveMetrics, setLiveMetrics] = useState({
    bankroll: 1250.75,
    sessionProfit: 250.75,
    winRate: 0.68,
    sharpeRatio: 1.42,
    currentRisk: 0.15,
    detectionRisk: 0.03
  })

  // Simulate live data updates
  useEffect(() => {
    const interval = setInterval(() => {
      setLiveMetrics(prev => ({
        ...prev,
        bankroll: prev.bankroll + (Math.random() - 0.5) * 10,
        sessionProfit: prev.sessionProfit + (Math.random() - 0.5) * 5,
        winRate: Math.max(0.4, Math.min(0.9, prev.winRate + (Math.random() - 0.5) * 0.02)),
        sharpeRatio: Math.max(0.5, Math.min(2.5, prev.sharpeRatio + (Math.random() - 0.5) * 0.1)),
        currentRisk: Math.max(0.05, Math.min(0.5, prev.currentRisk + (Math.random() - 0.5) * 0.02)),
        detectionRisk: Math.max(0.01, Math.min(0.2, prev.detectionRisk + (Math.random() - 0.5) * 0.01))
      }))
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  const strategies = [
    {
      id: 'takeshi',
      name: 'Takeshi',
      description: 'Aggressive Berserker',
      color: 'bg-red-500',
      icon: <Zap className="h-4 w-4" />,
      winRate: 0.52,
      avgReturn: -0.02,
      risk: 0.35
    },
    {
      id: 'lelouch',
      name: 'Lelouch',
      description: 'Strategic Mastermind',
      color: 'bg-teal-500',
      icon: <Brain className="h-4 w-4" />,
      winRate: 0.64,
      avgReturn: 0.015,
      risk: 0.18
    },
    {
      id: 'kazuya',
      name: 'Kazuya',
      description: 'Conservative Survivor',
      color: 'bg-blue-500',
      icon: <Shield className="h-4 w-4" />,
      winRate: 0.78,
      avgReturn: -0.005,
      risk: 0.08
    },
    {
      id: 'senku',
      name: 'Senku',
      description: 'Analytical Scientist',
      color: 'bg-green-500',
      icon: <Calculator className="h-4 w-4" />,
      winRate: 0.72,
      avgReturn: 0.025,
      risk: 0.15
    },
    {
      id: 'hybrid',
      name: 'Hybrid',
      description: 'Senku + Lelouch Fusion',
      color: 'bg-purple-500',
      icon: <Target className="h-4 w-4" />,
      winRate: 0.75,
      avgReturn: 0.032,
      risk: 0.12
    }
  ]

  const visualizations = [
    {
      category: 'Core Performance',
      items: [
        { name: 'Strategy Comparison', image: strategyComparison, description: 'Performance comparison across all strategies' },
        { name: 'Risk Analysis', image: riskAnalysis, description: 'Comprehensive risk assessment dashboard' },
        { name: 'Strategy Heatmap', image: strategyHeatmap, description: 'Performance across different game configurations' },
        { name: '3D Risk-Return Analysis', image: riskReturn3D, description: '3D visualization of risk vs return vs consistency' }
      ]
    },
    {
      category: 'Strategy Deep Dive',
      items: [
        { name: 'Takeshi Analysis', image: bankrollProgressionTakeshi, description: 'Aggressive strategy performance over time' },
        { name: 'Lelouch Analysis', image: bankrollProgressionLelouch, description: 'Strategic mastermind progression' },
        { name: 'Kazuya Analysis', image: bankrollProgressionKazuya, description: 'Conservative approach results' },
        { name: 'Senku Detailed', image: senkuDetailed, description: 'Comprehensive analytical dashboard' }
      ]
    },
    {
      category: 'Advanced Analytics',
      items: [
        { name: 'Parameter Sensitivity', image: parameterSensitivity, description: 'How parameters affect performance' },
        { name: 'Monte Carlo Results', image: monteCarloResults, description: 'Statistical simulation outcomes' },
        { name: 'Portfolio Optimization', image: portfolioOptimization, description: 'Multi-strategy portfolio analysis' },
        { name: 'Machine Learning', image: machineLearning, description: 'AI and adaptive algorithm performance' }
      ]
    },
    {
      category: 'Mathematical Foundations',
      items: [
        { name: 'Probability Theory', image: probabilityTheory, description: 'Mathematical foundations and theory' },
        { name: 'Game Mechanics', image: gameMechanics, description: 'Board analysis and optimal strategies' },
        { name: 'Detection Evasion', image: detectionEvasion, description: 'Human-like behavior patterns' },
        { name: 'Market Microstructure', image: marketMicrostructure, description: 'Platform and market analysis' }
      ]
    },
    {
      category: 'Specialized Analysis',
      items: [
        { name: 'Extreme Scenarios', image: extremeScenarios, description: 'Stress testing and black swan events' },
        { name: 'Behavioral Psychology', image: behavioralPsychology, description: 'Human factors and cognitive biases' },
        { name: 'Technical Implementation', image: technicalImplementation, description: 'System performance and architecture' }
      ]
    }
  ]

  const characterCards = [
    { name: 'Takeshi', image: characterCardTakeshi },
    { name: 'Lelouch', image: characterCardLelouch },
    { name: 'Kazuya', image: characterCardKazuya },
    { name: 'Senku', image: characterCardSenku }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm dark:bg-slate-900/80">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <GamepadIcon className="h-8 w-8 text-purple-600" />
                <div>
                  <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
                    Applied Probability Framework
                  </h1>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    High-RTP Games Automation & Analysis
                  </p>
                </div>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Badge variant="outline" className="text-green-600 border-green-600">
                <Activity className="h-3 w-3 mr-1" />
                Live
              </Badge>
              <Button 
                onClick={() => setSimulationRunning(!simulationRunning)}
                variant={simulationRunning ? "destructive" : "default"}
              >
                {simulationRunning ? "Stop Simulation" : "Start Simulation"}
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-6">
        <Tabs defaultValue="dashboard" className="space-y-6">
          <TabsList className="grid w-full grid-cols-6">
            <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
            <TabsTrigger value="strategies">Strategies</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
            <TabsTrigger value="visualizations">Visualizations</TabsTrigger>
            <TabsTrigger value="technical">Technical</TabsTrigger>
            <TabsTrigger value="reports">Reports</TabsTrigger>
          </TabsList>

          {/* Dashboard Tab */}
          <TabsContent value="dashboard" className="space-y-6">
            {/* Live Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Bankroll</CardTitle>
                  <DollarSign className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">${liveMetrics.bankroll.toFixed(2)}</div>
                  <p className="text-xs text-muted-foreground">
                    +{((liveMetrics.bankroll - 1000) / 1000 * 100).toFixed(1)}% from start
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Session Profit</CardTitle>
                  <TrendingUp className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-green-600">
                    +${liveMetrics.sessionProfit.toFixed(2)}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {(liveMetrics.sessionProfit / 1000 * 100).toFixed(1)}% return
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
                  <Target className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{(liveMetrics.winRate * 100).toFixed(1)}%</div>
                  <Progress value={liveMetrics.winRate * 100} className="mt-2" />
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Sharpe Ratio</CardTitle>
                  <BarChart3 className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{liveMetrics.sharpeRatio.toFixed(2)}</div>
                  <p className="text-xs text-muted-foreground">
                    Risk-adjusted return
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Current Risk</CardTitle>
                  <Activity className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{(liveMetrics.currentRisk * 100).toFixed(1)}%</div>
                  <Progress value={liveMetrics.currentRisk * 100} className="mt-2" />
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Detection Risk</CardTitle>
                  <Eye className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-green-600">
                    {(liveMetrics.detectionRisk * 100).toFixed(1)}%
                  </div>
                  <p className="text-xs text-muted-foreground">Very Low</p>
                </CardContent>
              </Card>
            </div>

            {/* Strategy Overview */}
            <Card>
              <CardHeader>
                <CardTitle>Active Strategy: Hybrid (Senku + Lelouch Fusion)</CardTitle>
                <CardDescription>
                  Combining analytical learning with strategic adaptation for optimal performance
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Analytical Weight</span>
                      <span className="text-sm">60%</span>
                    </div>
                    <Progress value={60} />
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Strategic Weight</span>
                      <span className="text-sm">40%</span>
                    </div>
                    <Progress value={40} />
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Confidence Level</span>
                      <span className="text-sm">87%</span>
                    </div>
                    <Progress value={87} />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Quick Visualizations */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Strategy Performance Comparison</CardTitle>
                </CardHeader>
                <CardContent>
                  <img 
                    src={strategyComparison} 
                    alt="Strategy Comparison" 
                    className="w-full h-auto rounded-lg"
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Risk Analysis Dashboard</CardTitle>
                </CardHeader>
                <CardContent>
                  <img 
                    src={riskAnalysis} 
                    alt="Risk Analysis" 
                    className="w-full h-auto rounded-lg"
                  />
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Strategies Tab */}
          <TabsContent value="strategies" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Strategy Selection */}
              <Card>
                <CardHeader>
                  <CardTitle>Strategy Selection</CardTitle>
                  <CardDescription>Choose your approach to the mines</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {strategies.map((strategy) => (
                    <div
                      key={strategy.id}
                      className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                        activeStrategy === strategy.id
                          ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                      onClick={() => setActiveStrategy(strategy.id)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <div className={`p-2 rounded-full ${strategy.color} text-white`}>
                            {strategy.icon}
                          </div>
                          <div>
                            <h3 className="font-semibold">{strategy.name}</h3>
                            <p className="text-sm text-muted-foreground">{strategy.description}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-sm font-medium">
                            {(strategy.winRate * 100).toFixed(0)}% Win Rate
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {strategy.avgReturn > 0 ? '+' : ''}{(strategy.avgReturn * 100).toFixed(1)}% Avg Return
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>

              {/* Character Cards */}
              <Card>
                <CardHeader>
                  <CardTitle>Strategy Character Cards</CardTitle>
                  <CardDescription>Anime-inspired strategy personas</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    {characterCards.map((character) => (
                      <div key={character.name} className="text-center">
                        <img 
                          src={character.image} 
                          alt={`${character.name} Character Card`}
                          className="w-full h-auto rounded-lg shadow-md"
                        />
                        <p className="mt-2 font-medium">{character.name}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Strategy Performance Details */}
            <Card>
              <CardHeader>
                <CardTitle>Detailed Strategy Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <img 
                  src={senkuDetailed} 
                  alt="Senku Detailed Analysis" 
                  className="w-full h-auto rounded-lg"
                />
              </CardContent>
            </Card>
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>3D Risk-Return Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <img 
                    src={riskReturn3D} 
                    alt="3D Risk Return Analysis" 
                    className="w-full h-auto rounded-lg"
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Parameter Sensitivity</CardTitle>
                </CardHeader>
                <CardContent>
                  <img 
                    src={parameterSensitivity} 
                    alt="Parameter Sensitivity Analysis" 
                    className="w-full h-auto rounded-lg"
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Monte Carlo Simulation Results</CardTitle>
                </CardHeader>
                <CardContent>
                  <img 
                    src={monteCarloResults} 
                    alt="Monte Carlo Results" 
                    className="w-full h-auto rounded-lg"
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Portfolio Optimization</CardTitle>
                </CardHeader>
                <CardContent>
                  <img 
                    src={portfolioOptimization} 
                    alt="Portfolio Optimization" 
                    className="w-full h-auto rounded-lg"
                  />
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Visualizations Tab */}
          <TabsContent value="visualizations" className="space-y-6">
            {visualizations.map((category) => (
              <Card key={category.category}>
                <CardHeader>
                  <CardTitle>{category.category}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {category.items.map((item) => (
                      <div key={item.name} className="space-y-2">
                        <img 
                          src={item.image} 
                          alt={item.name}
                          className="w-full h-auto rounded-lg shadow-sm border"
                        />
                        <div>
                          <h4 className="font-medium">{item.name}</h4>
                          <p className="text-sm text-muted-foreground">{item.description}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </TabsContent>

          {/* Technical Tab */}
          <TabsContent value="technical" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Machine Learning Performance</CardTitle>
                </CardHeader>
                <CardContent>
                  <img 
                    src={machineLearning} 
                    alt="Machine Learning Analysis" 
                    className="w-full h-auto rounded-lg"
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Detection Evasion Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <img 
                    src={detectionEvasion} 
                    alt="Detection Evasion" 
                    className="w-full h-auto rounded-lg"
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Technical Implementation</CardTitle>
                </CardHeader>
                <CardContent>
                  <img 
                    src={technicalImplementation} 
                    alt="Technical Implementation" 
                    className="w-full h-auto rounded-lg"
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>System Architecture</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                      <div className="flex items-center space-x-2">
                        <Cpu className="h-5 w-5 text-blue-500" />
                        <span className="font-medium">Java GUI Controller</span>
                      </div>
                      <Badge variant="outline" className="text-green-600">Active</Badge>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                      <div className="flex items-center space-x-2">
                        <Brain className="h-5 w-5 text-purple-500" />
                        <span className="font-medium">Python AI Engine</span>
                      </div>
                      <Badge variant="outline" className="text-green-600">Active</Badge>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                      <div className="flex items-center space-x-2">
                        <Shield className="h-5 w-5 text-green-500" />
                        <span className="font-medium">Detection Evasion</span>
                      </div>
                      <Badge variant="outline" className="text-green-600">Enabled</Badge>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                      <div className="flex items-center space-x-2">
                        <LineChart className="h-5 w-5 text-orange-500" />
                        <span className="font-medium">Real-time Analytics</span>
                      </div>
                      <Badge variant="outline" className="text-green-600">Running</Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Reports Tab */}
          <TabsContent value="reports" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Comprehensive Analysis Report</CardTitle>
                <CardDescription>
                  Full academic-grade analysis suitable for college applications and research
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  <Button variant="outline" className="h-20 flex flex-col items-center justify-center">
                    <PieChart className="h-6 w-6 mb-2" />
                    <span>Performance Report</span>
                  </Button>
                  <Button variant="outline" className="h-20 flex flex-col items-center justify-center">
                    <BarChart3 className="h-6 w-6 mb-2" />
                    <span>Risk Assessment</span>
                  </Button>
                  <Button variant="outline" className="h-20 flex flex-col items-center justify-center">
                    <Brain className="h-6 w-6 mb-2" />
                    <span>ML Analysis</span>
                  </Button>
                  <Button variant="outline" className="h-20 flex flex-col items-center justify-center">
                    <Calculator className="h-6 w-6 mb-2" />
                    <span>Mathematical Foundations</span>
                  </Button>
                  <Button variant="outline" className="h-20 flex flex-col items-center justify-center">
                    <Users className="h-6 w-6 mb-2" />
                    <span>Behavioral Analysis</span>
                  </Button>
                  <Button variant="outline" className="h-20 flex flex-col items-center justify-center">
                    <Settings className="h-6 w-6 mb-2" />
                    <span>Technical Documentation</span>
                  </Button>
                </div>

                <Separator />

                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Academic Layer Integration</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                      <h4 className="font-medium text-blue-900 dark:text-blue-100">Probability Theory</h4>
                      <p className="text-sm text-blue-700 dark:text-blue-300 mt-1">
                        Hypergeometric distributions, Kelly Criterion, and expected value calculations
                      </p>
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                      <h4 className="font-medium text-green-900 dark:text-green-100">Decision Theory</h4>
                      <p className="text-sm text-green-700 dark:text-green-300 mt-1">
                        Optimal decision making under uncertainty and risk management
                      </p>
                    </div>
                    <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                      <h4 className="font-medium text-purple-900 dark:text-purple-100">Machine Learning</h4>
                      <p className="text-sm text-purple-700 dark:text-purple-300 mt-1">
                        Reinforcement learning, Bayesian inference, and pattern recognition
                      </p>
                    </div>
                    <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                      <h4 className="font-medium text-orange-900 dark:text-orange-100">Behavioral Economics</h4>
                      <p className="text-sm text-orange-700 dark:text-orange-300 mt-1">
                        Cognitive biases, risk perception, and human decision-making patterns
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Extreme Scenarios and Specialized Analysis */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Extreme Scenarios Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <img 
                    src={extremeScenarios} 
                    alt="Extreme Scenarios" 
                    className="w-full h-auto rounded-lg"
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Behavioral Psychology</CardTitle>
                </CardHeader>
                <CardContent>
                  <img 
                    src={behavioralPsychology} 
                    alt="Behavioral Psychology" 
                    className="w-full h-auto rounded-lg"
                  />
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

export default App

