'use client'

import { useState, useCallback, useRef } from 'react'
import Link from 'next/link'
import { 
  Upload, 
  FileImage, 
  AlertTriangle, 
  CheckCircle, 
  XCircle, 
  Shield, 
  Eye,
  Clock,
  Brain,
  Info,
  Home,
  ExternalLink,
  Search,
  Sparkles
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Separator } from '@/components/ui/separator'
import { ThemeSwitcher } from '@/components/theme-switcher'
import { cn } from '@/lib/utils'

interface ReverseImageResult {
  url: string
  title: string
  source: string
}

interface AnalysisResult {
  verdict: string
  confidence: number
  probability: number
  explanation: string
  issuesFound?: string[]
  riskLevel: 'low' | 'medium' | 'high'
  details: {
    sharpness: number
    compressionArtifacts: number
    colorConsistency: number
    lightingConsistency: number
    facesDetected: number
  }
  similarImages?: ReverseImageResult[]
  reverseSearchUrl?: string
  processingTime: string
  filename: string
  filesize: string
}

export default function DeepfakeCheckPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = async (file: File) => {
    // Validate file type
    const supportedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
    if (!supportedTypes.includes(file.type)) {
      setError('Please upload a JPG, PNG, or WebP image')
      return
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB')
      return
    }

    setSelectedFile(file)
    setError(null)
    setAnalysis(null)

    // Create preview
    const reader = new FileReader()
    reader.onload = () => setPreviewUrl(reader.result as string)
    reader.readAsDataURL(file)
  }

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0])
    }
  }, [])

  const analyzeImage = async () => {
    if (!selectedFile) return

    setIsAnalyzing(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch('/api/deepfake/analyze-free', {
        method: 'POST',
        body: formData
      })

      const data = await response.json()

      if (!data.success) {
        throw new Error(data.error || 'Analysis failed')
      }

      setAnalysis(data.analysis)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const resetAnalysis = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setAnalysis(null)
    setError(null)
  }

  const getVerdictIcon = (verdict: string) => {
    if (verdict === 'likely authentic') return <CheckCircle className="h-8 w-8 text-green-500" />
    if (verdict === 'possibly manipulated') return <AlertTriangle className="h-8 w-8 text-yellow-500" />
    return <XCircle className="h-8 w-8 text-red-500" />
  }

  const getVerdictColor = (verdict: string) => {
    if (verdict === 'likely authentic') return 'text-green-500'
    if (verdict === 'possibly manipulated') return 'text-yellow-500'
    return 'text-red-500'
  }

  const getRiskBadgeVariant = (risk: string) => {
    if (risk === 'low') return 'secondary'
    if (risk === 'medium') return 'default'
    return 'destructive'
  }

  return (
    <div className="min-h-screen gradient-bg">
      {/* Header */}
      <div className="border-b glass sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link href="/">
                <Button variant="ghost" size="icon" title="Back to Home">
                  <Home className="h-5 w-5" />
                </Button>
              </Link>
              <div className="flex items-center space-x-2">
                <Shield className="h-8 w-8 text-primary" />
                <h1 className="text-2xl font-bold">Deepfake Detector</h1>
              </div>
              <Badge variant="secondary" className="hidden sm:inline-flex">
                <Sparkles className="h-3 w-3 mr-1" />
                100% Free & Open Source
              </Badge>
            </div>
            
            <div className="flex items-center space-x-2">
              <ThemeSwitcher />
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Info Banner */}
        <Alert className="mb-6">
          <Info className="h-4 w-4" />
          <AlertTitle>Free & Privacy-First</AlertTitle>
          <AlertDescription>
            This tool uses open-source AI models running locally. No data is sent to third-party services. 
            Results are for informational purposes only.
          </AlertDescription>
        </Alert>

        {/* Upload Area */}
        {!selectedFile && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Upload className="h-5 w-5" />
                <span>Upload Image</span>
              </CardTitle>
              <CardDescription>
                Upload an image to check for AI-generated content or manipulation
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div
                className={cn(
                  "relative border-2 border-dashed rounded-lg p-12 text-center transition-all duration-200",
                  dragActive 
                    ? "border-primary bg-primary/5 scale-[1.02]" 
                    : "border-muted-foreground/25 hover:border-muted-foreground/50",
                  "group cursor-pointer"
                )}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/jpg,image/png,image/webp"
                  onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
                  className="hidden"
                />
                
                <div className="space-y-4">
                  <div className="mx-auto w-20 h-20 bg-primary/10 rounded-full flex items-center justify-center group-hover:bg-primary/20 transition-colors">
                    <FileImage className="h-10 w-10 text-primary" />
                  </div>
                  
                  <div>
                    <p className="text-lg font-medium">Drop an image here or click to upload</p>
                    <p className="text-sm text-muted-foreground mt-1">
                      Supports JPG, PNG, WebP (max 10MB)
                    </p>
                  </div>
                </div>
              </div>

              {error && (
                <Alert variant="destructive" className="mt-4">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        )}

        {/* Analysis Section */}
        {selectedFile && (
          <div className="space-y-6">
            {/* Preview Card */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>Selected Image</CardTitle>
                  <Button variant="outline" size="sm" onClick={resetAnalysis}>
                    Upload Different Image
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {previewUrl && (
                  <div className="relative rounded-lg overflow-hidden bg-muted">
                    <img 
                      src={previewUrl} 
                      alt="Preview"
                      className="w-full max-h-96 object-contain"
                    />
                  </div>
                )}
                
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">{selectedFile.name}</span>
                  <span className="text-muted-foreground">
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </span>
                </div>

                {!analysis && !isAnalyzing && (
                  <Button 
                    className="w-full" 
                    size="lg"
                    onClick={analyzeImage}
                  >
                    <Brain className="h-4 w-4 mr-2" />
                    Analyze for Deepfakes
                  </Button>
                )}

                {isAnalyzing && (
                  <div className="text-center py-8">
                    <div className="animate-spin h-12 w-12 border-4 border-primary border-t-transparent rounded-full mx-auto mb-4" />
                    <p className="text-sm text-muted-foreground">Analyzing image...</p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Results Card */}
            {analysis && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Brain className="h-5 w-5" />
                    <span>Analysis Results</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Main Verdict */}
                  <div className="text-center space-y-4 py-6">
                    {getVerdictIcon(analysis.verdict)}
                    <div>
                      <h3 className={cn("text-2xl font-bold capitalize", getVerdictColor(analysis.verdict))}>
                        {analysis.verdict}
                      </h3>
                      <div className="flex items-center justify-center space-x-2 mt-2">
                        <Badge variant={getRiskBadgeVariant(analysis.riskLevel)}>
                          {analysis.riskLevel.toUpperCase()} RISK
                        </Badge>
                        <span className="text-sm text-muted-foreground">
                          {analysis.confidence}% confidence
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Probability Score Bar */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="font-medium">Manipulation Probability</span>
                      <span className="font-bold">{analysis.probability}%</span>
                    </div>
                    <Progress value={analysis.probability} className="h-3" />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Authentic</span>
                      <span>Manipulated</span>
                    </div>
                  </div>

                  <Separator />

                  {/* Explanation */}
                  <Alert>
                    <Eye className="h-4 w-4" />
                    <AlertTitle>Why this verdict?</AlertTitle>
                    <AlertDescription>{analysis.explanation}</AlertDescription>
                  </Alert>

                  {/* Detailed Issues Found */}
                  {analysis.issuesFound && analysis.issuesFound.length > 0 && (
                    <div className="space-y-2">
                      <h4 className="font-medium text-sm flex items-center space-x-2">
                        <AlertTriangle className="h-4 w-4" />
                        <span>Specific Issues Detected:</span>
                      </h4>
                      <ul className="space-y-1 text-sm text-muted-foreground">
                        {analysis.issuesFound.map((issue, idx) => (
                          <li key={idx} className="flex items-start space-x-2">
                            <span className="text-destructive mt-0.5">•</span>
                            <span>{issue}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Technical Details */}
                  <div className="space-y-3">
                    <h4 className="font-medium text-sm">Technical Metrics</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">Sharpness</span>
                          <span className="font-medium">{analysis.details.sharpness.toFixed(1)}</span>
                        </div>
                        <Progress value={analysis.details.sharpness} className="h-1" />
                      </div>
                      
                      <div className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">Color Consistency</span>
                          <span className="font-medium">{analysis.details.colorConsistency.toFixed(1)}</span>
                        </div>
                        <Progress value={analysis.details.colorConsistency} className="h-1" />
                      </div>
                      
                      <div className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">Lighting Analysis</span>
                          <span className="font-medium">{analysis.details.lightingConsistency.toFixed(1)}</span>
                        </div>
                        <Progress value={analysis.details.lightingConsistency} className="h-1" />
                      </div>
                      
                      <div className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">Compression Artifacts</span>
                          <span className="font-medium">{analysis.details.compressionArtifacts.toFixed(1)}</span>
                        </div>
                        <Progress value={analysis.details.compressionArtifacts} className="h-1" />
                      </div>
                    </div>
                    
                    <div className="text-xs text-muted-foreground pt-2">
                      Faces detected: {analysis.details.facesDetected} • Processing time: {analysis.processingTime}
                    </div>
                  </div>

                  {/* Reverse Image Search */}
                  {analysis.similarImages && analysis.similarImages.length > 0 && (
                    <>
                      <Separator />
                      <div className="space-y-3">
                        <h4 className="font-medium text-sm flex items-center space-x-2">
                          <Search className="h-4 w-4" />
                          <span>Verify with Reverse Image Search</span>
                        </h4>
                        <Alert>
                          <Info className="h-4 w-4" />
                          <AlertDescription className="text-xs">
                            Upload your image to these search engines to find if it appears elsewhere online or has been manipulated.
                          </AlertDescription>
                        </Alert>
                        <div className="space-y-2">
                          {analysis.similarImages.map((result, idx) => (
                            <a
                              key={idx}
                              href={result.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="block p-3 rounded-lg border hover:bg-muted/50 transition-colors"
                            >
                              <div className="flex items-center justify-between">
                                <div className="flex-1 min-w-0">
                                  <p className="text-sm font-medium truncate">{result.title}</p>
                                  <p className="text-xs text-muted-foreground">{result.source}</p>
                                </div>
                                <ExternalLink className="h-4 w-4 ml-2 text-muted-foreground flex-shrink-0" />
                              </div>
                            </a>
                          ))}
                        </div>
                        {previewUrl && (
                          <p className="text-xs text-muted-foreground">
                            💡 Tip: Save your image first, then drag & drop it into these search engines
                          </p>
                        )}
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>
            )}
          </div>
        )}

        {/* How it Works */}
        {!selectedFile && (
          <Card className="mt-6">
            <CardHeader>
              <CardTitle className="text-lg">How It Works</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm text-muted-foreground">
              <div className="flex items-start space-x-3">
                <div className="bg-primary/10 rounded-full p-2 mt-0.5">
                  <Brain className="h-4 w-4 text-primary" />
                </div>
                <div>
                  <p className="font-medium text-foreground">AI Analysis</p>
                  <p>Uses computer vision and deep learning to detect manipulation artifacts</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="bg-primary/10 rounded-full p-2 mt-0.5">
                  <Shield className="h-4 w-4 text-primary" />
                </div>
                <div>
                  <p className="font-medium text-foreground">Privacy First</p>
                  <p>All processing happens on our servers - no third-party APIs or data sharing</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <div className="bg-primary/10 rounded-full p-2 mt-0.5">
                  <Search className="h-4 w-4 text-primary" />
                </div>
                <div>
                  <p className="font-medium text-foreground">Reverse Image Search</p>
                  <p>Optional links to find similar images and verify authenticity</p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

