import React, { useState } from "react";
import { ChevronLeft, ChevronRight, Database, TrendingUp, ShoppingCart, MessageSquare, Users, BarChart3, AlertCircle, CheckCircle } from 'lucide-react';

const Presentation = () => {
  const [currentSlide, setCurrentSlide] = useState(0);

  const slides = [
    // Slide 0: Title
    {
      type: 'title',
      title: '카페 ERP/마케팅 보조를 위한 GCP 기반 대시보드 설계와 프로토타입 연구',
      subtitle: 'A Study on the Design and Prototyping of a GCP-Based Dashboard for Small and Medium-sized Café Businesses',
      authors: [
        '이재호* · 이서진** · 노현호** · 이한진***',
        '*한동대학교 AI융합학부',
        '**한동대학교 경영경제학부',
        '***한동대학교 창의융합교육원'
      ],
      conference: '한국멀티미디어학회 춘계학술발표대회 2025'
    },
    // Slide 1: Research Question
    {
      type: 'question',
      title: '핵심 질문',
      question: '소상공인 카페용 ERP·마케팅 대시보드\n초기 도입 성공을 가장 크게 좌우하는 것은?',
      options: [
        { letter: 'A', text: '기능 개수 (기능이 많을수록 좋다)', correct: false },
        { letter: 'B', text: '완전 자동화 (사람 개입 0)', correct: false },
        { letter: 'C', text: '입력 시간 단축 · 익숙한 도구와의 유사성', correct: true },
        { letter: 'D', text: '최고급 ML 모델 적용', correct: false }
      ],
      answer: '정답: C - 현장 인터뷰 결과, 입력 시간과 익숙한 도구(엑셀·카톡)와의 유사성이 수용성의 핵심 (3개 매장 공통)'
    },
    // Slide 2: Contents
    {
      type: 'contents',
      title: '목차',
      items: [
        { num: '01', text: '연구 배경 및 목적', icon: AlertCircle },
        { num: '02', text: '선행연구 및 현행 방식의 한계', icon: Database },
        { num: '03', text: '연구 방법론', icon: Users },
        { num: '04', text: '시스템 설계 및 구현', icon: BarChart3 },
        { num: '05', text: '시사점 및 결론', icon: CheckCircle }
      ]
    },
    // Slide 3: Background
    {
      type: 'content',
      title: '01. 연구 배경 및 목적',
      sections: [
        {
          subtitle: '문제 상황',
          points: [
            '다채널 운영 (POS·배달앱·스마트스토어) → 데이터 사일로화',
            '수기 입력·중복 관리 → 운영 피로도 증가 및 비용 상승',
            '실시간 재고·원가 파악 어려움 → 결품 위험 증가'
          ]
        },
        {
          subtitle: '연구 목표',
          points: [
            '재고·원가·매출·마케팅을 통합한 단일 대시보드 설계',
            'GCP 기반 경량 스택으로 저비용·빠른 구축 실현',
            '현장 검증을 통한 실용적 프로토타입 개발'
          ]
        }
      ]
    },
    // Slide 4: Prior Research
    {
      type: 'research',
      title: '02. 선행연구',
      findings: [
        {
          topic: 'ERP 도입 성공 요인',
          insight: '기술보다 사전 프로세스 표준화·데이터 거버넌스가 핵심',
          source: 'Christofi et al., 2013'
        },
        {
          topic: 'SME 업무 통합',
          insight: 'ERP·KM·AI 통합이 신속한 의사결정과 자동화에 기여',
          source: 'Weli et al., 2024'
        },
        {
          topic: '데이터 사일로 문제',
          insight: '연결성 부재는 생산성 저하·중복 작업 유발',
          source: 'Salesforce Research, 2025'
        },
        {
          topic: '마케팅 워크플로우',
          insight: '생성형 AI를 팀 기반 마케팅에 안전하게 접목',
          source: 'Nguyen & Miller, 2025'
        }
      ]
    },
    // Slide 5: Limitations
    {
      type: 'split',
      title: '현행 방식의 한계',
      left: {
        subtitle: '기술적 한계',
        items: [
          { icon: '💰', text: '고비용 SaaS 솔루션', detail: '소규모 매장 도입 장벽' },
          { icon: '📊', text: '데이터 사일로', detail: 'POS·마케팅·재고 분리' },
          { icon: '⚙️', text: '과도한 자동화', detail: '검토 없이 실행하는 위험' }
        ]
      },
      right: {
        subtitle: '운영적 한계',
        items: [
          { icon: '✍️', text: '수기 입력 병행', detail: '엑셀·카톡·POS 중복 관리' },
          { icon: '🔍', text: '추적성 부재', detail: '재고 이력 관리 미흡' },
          { icon: '📱', text: 'UX 불일치', detail: '익숙한 도구와 괴리감' }
        ]
      }
    },
    // Slide 6: Methodology
    {
      type: 'methodology',
      title: '03. 연구 방법론',
      methods: [
        {
          step: '1',
          title: '현장 인터뷰',
          desc: '3개 카페 매장 대상 반구조화 인터뷰',
          details: ['프랜차이즈 본사형', '개인 운영형', '가족 운영형']
        },
        {
          step: '2',
          title: '요구사항 도출',
          desc: '공통 페인포인트 및 핵심 니즈 분석',
          details: ['데이터 통합', '입력 부담', '비용 장벽']
        },
        {
          step: '3',
          title: '시스템 설계',
          desc: 'Firestore-BigQuery-Streamlit 아키텍처',
          details: ['반자동 보조', '표준 스키마', '단일 화면']
        },
        {
          step: '4',
          title: '프로토타입 구현',
          desc: 'Kaggle Coffee Shop Sales 데이터 활용',
          details: ['2023.01-06 → 2025 변환', 'BOM 자동 차감']
        }
      ]
    },
    // Slide 7: Interview Insights
    {
      type: 'insights',
      title: '현장 인터뷰 핵심 인사이트',
      insights: [
        {
          category: '공통 Pain Points',
          items: [
            '데이터 통합 부재 - 채널별 데이터 수동 취합',
            '수기 입력 부담 - 중복 작업으로 인한 피로도',
            '시스템 비용 장벽 - 월 구독료 부담',
            '완전 자동화 거부감 - 검토 없는 실행 불안'
          ]
        },
        {
          category: '핵심 요구사항',
          items: [
            '모바일 접근 가능 - 현장에서 즉시 확인',
            '저비용 경량 솔루션 - 단계적 확장 가능',
            '표준 데이터 스키마 - 일관된 관리',
            '익숙한 UX - 엑셀/카톡 유사성'
          ]
        }
      ],
      quote: '"하루 30분 이상 걸리던 재고 입력을 10분 이내로 줄일 수 있다면 당장 도입하겠다"'
    },
    // Slide 8: System Architecture
    {
      type: 'architecture',
      title: '04. 시스템 아키텍처',
      layers: [
        { name: 'Data Layer', tech: 'Firestore', desc: '운영 데이터 저장', icon: Database },
        { name: 'Analytics Layer', tech: 'BigQuery', desc: '일/주/월 집계 및 분석', icon: BarChart3 },
        { name: 'Presentation Layer', tech: 'Streamlit', desc: '대시보드 UI', icon: TrendingUp }
      ],
      flow: [
        'POS/배달앱 데이터 → Firestore 실시간 저장',
        '판매 기록 → BOM 기반 자동 재고 차감',
        'BigQuery 집계 → KPI/분석/운영 탭 표시'
      ]
    },
    // Slide 9: Data Model
    {
      type: 'datamodel',
      title: '데이터 모델 및 핵심 로직',
      collections: [
        { name: 'coffee_sales', desc: '판매 거래 기록', fields: 'transaction_id, product_name, quantity, price' },
        { name: 'recipes', desc: 'BOM (Bill of Materials)', fields: 'product_name, ingredient_name, quantity' },
        { name: 'inventory', desc: '재고 현황', fields: 'ingredient_name, current_qty, unit' },
        { name: 'stock_moves', desc: '재고 이동 로그', fields: 'ingredient_name, move_type, qty_before, qty_after' }
      ],
      logic: [
        { formula: '필요량', calc: '판매수량 × 레시피수량 × (1 + 폐기율)' },
        { formula: 'ROP', calc: '일평균소진 × 리드타임 + 안전재고' },
        { formula: '권장발주', calc: '목표일수 × 일평균소진 - 현재재고' }
      ]
    },
    // Slide 10: UI Dashboard
    {
      type: 'dashboard',
      title: 'UI/대시보드 구성',
      tabs: [
        { name: 'KPI', icon: TrendingUp, features: ['일/주/월 매출', '마진율', '결품 위험 지표', '목표 대비 달성률'] },
        { name: '분석', icon: BarChart3, features: ['품목별 매출 TOP-N', '시계열 추세', '마진 기여도', '카테고리별 분석'] },
        { name: '운영', icon: ShoppingCart, features: ['실시간 재고 현황', 'ROP 경보', '권장 발주량', '입고/이동 기록'] },
        { name: '마케팅', icon: MessageSquare, features: ['간단 CRM (베타)', '쿠폰 관리', '메시지 템플릿', '리텐션 지표'] }
      ],
      principle: '단일 화면 · 한글 중심 UI · 탭 전환 방식'
    },
    // Slide 11: Implementation Details
    {
      type: 'implementation',
      title: '구현 세부사항',
      features: [
        {
          title: '반자동 보조 시스템',
          desc: '거래 저장 시 BOM 기반 자동 차감 (기본값)',
          detail: '수정/삭제 시 재고 반영 여부 선택 가능'
        },
        {
          title: '추적성 보장',
          desc: '모든 재고 증감은 stock_moves에 로그',
          detail: '시각·유형·전후 잔량 기록'
        },
        {
          title: '단위 표준화',
          desc: 'g/kg, ml/l, ea 정규화',
          detail: '밀도 미지시 g↔ml 변환 금지 (보수적)'
        },
        {
          title: '운영 가드레일',
          desc: 'Cloud Logging/Monitoring + Budget Alerts',
          detail: '50%/80%/100% 임계값 알림'
        }
      ]
    },
    // Slide 12: Tech Stack
    {
      type: 'techstack',
      title: 'GCP 기술 스택 및 배포',
      stack: [
        { tech: 'Firestore', use: '운영 데이터 저장 (NoSQL)', cost: '1GB 무료 → ~$0.18/GB' },
        { tech: 'BigQuery', use: '분석 쿼리 (SQL)', cost: '1TB 쿼리 무료 → $5/TB' },
        { tech: 'Streamlit', use: 'Python 기반 대시보드', cost: 'Open Source (무료)' },
        { tech: 'Cloud Run', use: '컨테이너 배포 (예정)', cost: '월 200만 요청 무료' }
      ],
      security: [
        'Service Account 최소 권한 (IAM)',
        'Secret Manager (API Key 관리)',
        'VPC 방화벽 규칙'
      ],
      deployment: '현재: 로컬/개발 환경 | 향후: Cloud Run 경량 배포'
    },
    // Slide 13: Demo Results
    {
      type: 'demo',
      title: '프로토타입 시연 결과',
      scenario: 'Kaggle Coffee Shop Sales 데이터 (2023.01-06 → 2025 변환)',
      results: [
        { metric: '데이터 처리', value: '149,116건 거래', status: 'success' },
        { metric: 'BOM 매칭', value: '37개 제품 → 레시피 연결', status: 'success' },
        { metric: '재고 차감', value: '실시간 자동 차감', status: 'success' },
        { metric: 'ROP 경보', value: '12개 품목 임계 도달', status: 'warning' },
        { metric: '권장 발주', value: '자동 계산 및 표시', status: 'success' }
      ],
      validation: '입력 → 차감 → 경보 → 권장 발주가 단일 화면에서 작동 확인'
    },
    // Slide 14: Implications
    {
      type: 'implications',
      title: '05. 시사점',
      points: [
        {
          title: '현실 친화적 도입 전략',
          desc: '"반자동 보조 + 단일 화면 + 한글 고정 표기"가 초기 수용성 극대화',
          impact: '완전 자동화보다 검토 가능한 보조가 신뢰 확보'
        },
        {
          title: '데이터 표준화의 가치',
          desc: 'sales → recipes → inventory 표준 키·단위 정규화',
          impact: '오류·중복 감소 및 확장성 확보'
        },
        {
          title: '경량 스택의 효율성',
          desc: 'Firestore-BigQuery-Streamlit 조합',
          impact: '저비용·빠른 검증(MVP) 가능, 월 $50 이하 운영'
        },
        {
          title: '운영 가드레일 선행',
          desc: '관측/비용 가드레일 우선 구축',
          impact: '문제 발생 시 대응 속도 향상'
        }
      ]
    },
    // Slide 15: Future Work
    {
      type: 'future',
      title: '후속 연구',
      tracks: [
        {
          category: '시스템 고도화',
          items: [
            'POS/스마트스토어/배달앱 실시간 연동',
            '로우데이터 표준 스키마 정의',
            'Cloud Run 정식 배포 및 CI/CD 파이프라인'
          ]
        },
        {
          category: 'UX 개선',
          items: [
            '확정 전 검토 흐름 강화',
            '입력 시간 추가 단축 (음성 입력 검토)',
            '모바일 네이티브 앱 개발'
          ]
        },
        {
          category: '기능 확장',
          items: [
            '경량 CRM 베타 (메시지·쿠폰·리텐션)',
            '예측 모델 (수요 예측·최적 발주)',
            '다매장 통합 관리 (프랜차이즈)'
          ]
        },
        {
          category: '효과 검증',
          items: [
            '입력 시간 단축 정량 측정',
            '결품률 감소 측정',
            '마케팅 콘텐츠 리드타임 측정'
          ]
        }
      ]
    },
    // Slide 16: Conclusion
    {
      type: 'conclusion',
      title: '결론',
      contributions: [
        'GCP 경량 스택으로 카페 ERP/마케팅 보조 통합 대시보드의 설계·프로토타입 제시',
        '3개 매장 인터뷰를 통한 현장 검증 및 실용적 인사이트 도출',
        'Firestore-BigQuery-Streamlit 아키텍처로 저비용·빠른 구축 실증'
      ],
      keyFindings: [
        'BOM 기반 자동 재고 차감으로 수기 입력 부담 경감',
        'ROP/권장 발주 자동 계산으로 결품 위험 최소화',
        '표준화된 데이터 흐름으로 확장성 확보'
      ],
      message: '"작게 시작하되 체계적으로 확장"',
      tagline: 'URL로 검증 가능한 운영형 실험체를 빠르게 확보하라'
    },
    // Slide 17: References
    {
      type: 'references',
      title: '참고문헌',
      refs: [
        '[1] Christofi, M., Leonidou, E., & Vrontis, D. (2013). ERP implementation success factors in SMEs. Journal of Business Research.',
        '[2] Weli, W., Rorimpandey, L., & Wowor, J. (2024). Integration of ERP, Knowledge Management, and AI in SME Accounting Systems. International Journal of Business Technology.',
        '[3] Nguyen, T., & Miller, J. (2025). Generative AI in Team-Based Marketing Workflows. Marketing Science Quarterly.',
        '[4] Salesforce Research. (2025). The State of Data Silos: Impact on Business Productivity. Salesforce Research Report.',
        '[5] Ibrahim, A. (2023). Coffee Shop Sales Dataset. Kaggle. https://www.kaggle.com/datasets/ahmedabbas757/coffee-sales'
      ]
    },
    // Slide 18: Q&A
    {
      type: 'qa',
      title: 'Q & A',
      contact: {
        email: 'discover@handong.ac.kr',
        team: 'GCP ERP 개발 팀',
        institution: '한동대학교'
      }
    }
  ];

  const nextSlide = () => setCurrentSlide((prev) => Math.min(prev + 1, slides.length - 1));
  const prevSlide = () => setCurrentSlide((prev) => Math.max(prev - 1, 0));

  const renderSlide = (slide) => {
    switch(slide.type) {
      case 'title':
        return (
          <div className="flex flex-col items-center justify-center h-full bg-gradient-to-br from-blue-600 to-blue-800 text-white p-12">
            <h1 className="text-4xl font-bold text-center mb-6 leading-tight">{slide.title}</h1>
            <p className="text-xl text-center mb-12 text-blue-100">{slide.subtitle}</p>
            <div className="space-y-2 text-center">
              {slide.authors.map((author, i) => (
                <p key={i} className="text-lg text-blue-100">{author}</p>
              ))}
            </div>
            <div className="mt-12 pt-8 border-t border-blue-400">
              <p className="text-xl font-semibold">{slide.conference}</p>
            </div>
          </div>
        );

      case 'question':
        return (
          <div className="p-12 h-full bg-gradient-to-br from-purple-50 to-blue-50">
            <h2 className="text-3xl font-bold text-gray-800 mb-8">{slide.title}</h2>
            <div className="bg-white rounded-lg p-8 shadow-lg mb-8">
              <p className="text-2xl font-semibold text-center text-gray-800 mb-6 whitespace-pre-line">{slide.question}</p>
            </div>
            <div className="space-y-4 mb-8">
              {slide.options.map((opt) => (
                <div key={opt.letter} className={`p-4 rounded-lg border-2 ${opt.correct ? 'bg-green-50 border-green-500' : 'bg-white border-gray-300'}`}>
                  <span className="font-bold text-lg">{opt.letter}.</span> <span className="text-lg">{opt.text}</span>
                </div>
              ))}
            </div>
            <div className="bg-green-100 border-l-4 border-green-500 p-4 rounded">
              <p className="text-lg font-semibold text-green-800">{slide.answer}</p>
            </div>
          </div>
        );

      case 'contents':
        return (
          <div className="p-12 h-full bg-gray-50">
            <h2 className="text-4xl font-bold text-gray-800 mb-12">{slide.title}</h2>
            <div className="grid grid-cols-1 gap-6">
              {slide.items.map((item, i) => {
                const Icon = item.icon;
                return (
                  <div key={i} className="flex items-center bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition">
                    <div className="bg-blue-100 p-4 rounded-lg mr-6">
                      <Icon className="w-8 h-8 text-blue-600" />
                    </div>
                    <div>
                      <span className="text-2xl font-bold text-blue-600 mr-4">{item.num}</span>
                      <span className="text-2xl font-semibold text-gray-800">{item.text}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );

      case 'content':
        return (
          <div className="p-12 h-full bg-white overflow-y-auto">
            <h2 className="text-3xl font-bold text-gray-800 mb-8">{slide.title}</h2>
            {slide.sections.map((section, i) => (
              <div key={i} className="mb-8">
                <h3 className="text-2xl font-semibold text-blue-600 mb-4">{section.subtitle}</h3>
                <ul className="space-y-3">
                  {section.points.map((point, j) => (
                    <li key={j} className="flex items-start">
                      <span className="text-blue-500 mr-3 mt-1">▪</span>
                      <span className="text-lg text-gray-700">{point}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        );

      case 'research':
        return (
          <div className="p-12 h-full bg-gradient-to-br from-blue-50 to-purple-50">
            <h2 className="text-3xl font-bold text-gray-800 mb-8">{slide.title}</h2>
            <div className="grid grid-cols-1 gap-4">
              {slide.findings.map((finding, i) => (
                <div key={i} className="bg-white p-6 rounded-lg shadow-md border-l-4 border-blue-500">
                  <h3 className="text-xl font-bold text-gray-800 mb-2">{finding.topic}</h3>
                  <p className="text-lg text-gray-700 mb-2">{finding.insight}</p>
                  <p className="text-sm text-gray-500 italic">— {finding.source}</p>
                </div>
              ))}
            </div>
          </div>
        );

      case 'split':
        return (
          <div className="p-12 h-full bg-white">
            <h2 className="text-3xl font-bold text-gray-800 mb-8 text-center">{slide.title}</h2>
            <div className="grid grid-cols-2 gap-8 h-4/5">
              <div className="bg-red-50 p-6 rounded-lg">
                <h3 className="text-2xl font-bold text-red-600 mb-6">{slide.left.subtitle}</h3>
                <div className="space-y-4">
                  {slide.left.items.map((item, i) => (
                    <div key={i} className="bg-white p-4 rounded-lg shadow">
                      <div className="flex items-center mb-2">
                        <span className="text-3xl mr-3">{item.icon}</span>
                        <span className="text-lg font-semibold">{item.text}</span>
                      </div>
                      <p className="text-sm text-gray-600 ml-12">{item.detail}</p>
                    </div>
                  ))}
                </div>
              </div>
              <div className="bg-blue-50 p-6 rounded-lg">
                <h3 className="text-2xl font-bold text-blue-600 mb-6">{slide.right.subtitle}</h3>
                <div className="space-y-4">
                  {slide.right.items.map((item, i) => (
                    <div key={i} className="bg-white p-4 rounded-lg shadow">
                      <div className="flex items-center mb-2">
                        <span className="text-3xl mr-3">{item.icon}</span>
                        <span className="text-lg font-semibold">{item.text}</span>
                      </div>
                      <p className="text-sm text-gray-600 ml-12">{item.detail}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        );

      case 'methodology':
        return (
          <div className="p-12 h-full bg-gradient-to-br from-green-50 to-blue-50">
            <h2 className="text-3xl font-bold text-gray-800 mb-8">{slide.title}</h2>
            <div className="grid grid-cols-2 gap-6">
              {slide.methods.map((method) => (
                <div key={method.step} className="bg-white p-6 rounded-lg shadow-lg">
                  <div className="flex items-center mb-4">
                    <div className="bg-blue-600 text-white w-12 h-12 rounded-full flex items-center justify-center text-xl font-bold mr-4">
                      {method.step}
                    </div>
                    <h3 className="text-xl font-bold text-gray-800">{method.title}</h3>
                  </div>
                  <p className="text-lg text-gray-700 mb-3">{method.desc}</p>
                  <ul className="space-y-1">
                    {method.details.map((detail, i) => (
                      <li key={i} className="text-sm text-gray-600 ml-4">• {detail}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        );

      case 'insights':
        return (
          <div className="p-12 h-full bg-white overflow-y-auto">
            <h2 className="text-3xl font-bold text-gray-800 mb-8">{slide.title}</h2>
            {slide.insights.map((insight, i) => (
              <div key={i} className="mb-6">
                <h3 className="text-2xl font-semibold text-blue-600 mb-4 border-b-2 border-blue-200 pb-2">{insight.category}</h3>
                <ul className="space-y-2">
                  {insight.items.map((item, j) => (
                    <li key={j} className="flex items-start">
                      <CheckCircle className="w-5 h-5 text-green-500 mr-3 mt-1 flex-shrink-0" />
                      <span className="text-lg text-gray-700">{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
            <div className="mt-8 bg-blue-50 border-l-4 border-blue-500 p-6 rounded">
              <p className="text-xl italic text-gray-800">{slide.quote}</p>
              <p className="text-sm text-gray-600 mt-2">— 인터뷰 참여 매장주</p>
            </div>
          </div>
        );

      case 'architecture':
        return (
          <div className="p-12 h-full bg-gradient-to-br from-purple-50 to-blue-50">
            <h2 className="text-3xl font-bold text-gray-800 mb-8">{slide.title}</h2>
            <div className="grid grid-cols-3 gap-6 mb-8">
              {slide.layers.map((layer, i) => {
                const Icon = layer.icon;
                return (
                  <div key={i} className="bg-white p-6 rounded-lg shadow-lg text-center">
                    <div className="flex justify-center mb-4">
                      <div className="bg-blue-100 p-4 rounded-full">
                        <Icon className="w-10 h-10 text-blue-600" />
                      </div>
                    </div>
                    <h3 className="text-xl font-bold text-gray-800 mb-2">{layer.name}</h3>
                    <p className="text-lg font-semibold text-blue-600 mb-2">{layer.tech}</p>
                    <p className="text-sm text-gray-600">{layer.desc}</p>
                  </div>
                );
              })}
            </div>
            <div className="bg-white p-6 rounded-lg shadow-lg">
              <h3 className="text-xl font-bold text-gray-800 mb-4">데이터 흐름</h3>
              {slide.flow.map((step, i) => (
                <div key={i} className="flex items-center mb-3">
                  <div className="bg-blue-500 text-white w-8 h-8 rounded-full flex items-center justify-center mr-4 text-sm font-bold">
                    {i + 1}
                  </div>
                  <p className="text-lg text-gray-700">{step}</p>
                </div>
              ))}
            </div>
          </div>
        );

      case 'datamodel':
        return (
          <div className="p-12 h-full bg-white overflow-y-auto">
            <h2 className="text-3xl font-bold text-gray-800 mb-8">{slide.title}</h2>
            <div className="grid grid-cols-2 gap-6 mb-8">
              <div>
                <h3 className="text-xl font-bold text-gray-800 mb-4">Firestore Collections</h3>
                {slide.collections.map((col, i) => (
                  <div key={i} className="bg-blue-50 p-4 rounded-lg mb-3 border-l-4 border-blue-500">
                    <h4 className="font-bold text-lg text-gray-800">{col.name}</h4>
                    <p className="text-sm text-gray-600 mb-1">{col.desc}</p>
                    <p className="text-xs text-gray-500 font-mono">{col.fields}</p>
                  </div>
                ))}
              </div>
              <div>
                <h3 className="text-xl font-bold text-gray-800 mb-4">핵심 계산 로직</h3>
                {slide.logic.map((logic, i) => (
                  <div key={i} className="bg-green-50 p-4 rounded-lg mb-3 border-l-4 border-green-500">
                    <h4 className="font-bold text-lg text-gray-800 mb-2">{logic.formula}</h4>
                    <p className="text-base font-mono text-gray-700 bg-white p-2 rounded">{logic.calc}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );

      case 'dashboard':
        return (
          <div className="p-12 h-full bg-gradient-to-br from-blue-50 to-purple-50">
            <h2 className="text-3xl font-bold text-gray-800 mb-6">{slide.title}</h2>
            <div className="grid grid-cols-2 gap-6 mb-6">
              {slide.tabs.map((tab, i) => {
                const Icon = tab.icon;
                return (
                  <div key={i} className="bg-white p-6 rounded-lg shadow-lg">
                    <div className="flex items-center mb-4">
                      <Icon className="w-8 h-8 text-blue-600 mr-3" />
                      <h3 className="text-xl font-bold text-gray-800">{tab.name} 탭</h3>
                    </div>
                    <ul className="space-y-2">
                      {tab.features.map((feature, j) => (
                        <li key={j} className="flex items-start">
                          <span className="text-blue-500 mr-2">✓</span>
                          <span className="text-base text-gray-700">{feature}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                );
              })}
            </div>
            <div className="bg-blue-600 text-white p-4 rounded-lg text-center">
              <p className="text-xl font-semibold">{slide.principle}</p>
            </div>
          </div>
        );

      case 'implementation':
        return (
          <div className="p-12 h-full bg-white">
            <h2 className="text-3xl font-bold text-gray-800 mb-8">{slide.title}</h2>
            <div className="grid grid-cols-2 gap-6">
              {slide.features.map((feature, i) => (
                <div key={i} className="bg-gradient-to-br from-blue-50 to-purple-50 p-6 rounded-lg shadow-md">
                  <h3 className="text-xl font-bold text-gray-800 mb-3">{feature.title}</h3>
                  <p className="text-lg text-gray-700 mb-2">{feature.desc}</p>
                  <p className="text-sm text-gray-600 italic bg-white p-3 rounded">{feature.detail}</p>
                </div>
              ))}
            </div>
          </div>
        );

      case 'techstack':
        return (
          <div className="p-12 h-full bg-gradient-to-br from-gray-50 to-blue-50 overflow-y-auto">
            <h2 className="text-3xl font-bold text-gray-800 mb-6">{slide.title}</h2>
            <div className="grid grid-cols-2 gap-6 mb-6">
              <div>
                <h3 className="text-xl font-bold text-gray-800 mb-4">기술 스택</h3>
                {slide.stack.map((item, i) => (
                  <div key={i} className="bg-white p-4 rounded-lg shadow mb-3">
                    <h4 className="font-bold text-lg text-blue-600">{item.tech}</h4>
                    <p className="text-sm text-gray-700 mb-1">{item.use}</p>
                    <p className="text-xs text-green-600 font-semibold">{item.cost}</p>
                  </div>
                ))}
              </div>
              <div>
                <h3 className="text-xl font-bold text-gray-800 mb-4">보안 및 권한</h3>
                <div className="bg-white p-6 rounded-lg shadow mb-4">
                  {slide.security.map((sec, i) => (
                    <div key={i} className="flex items-center mb-3">
                      <CheckCircle className="w-5 h-5 text-green-500 mr-3" />
                      <span className="text-base text-gray-700">{sec}</span>
                    </div>
                  ))}
                </div>
                <div className="bg-blue-100 p-4 rounded-lg border-l-4 border-blue-500">
                  <p className="text-sm font-semibold text-gray-800">{slide.deployment}</p>
                </div>
              </div>
            </div>
          </div>
        );

      case 'demo':
        return (
          <div className="p-12 h-full bg-white">
            <h2 className="text-3xl font-bold text-gray-800 mb-6">{slide.title}</h2>
            <div className="bg-blue-50 p-4 rounded-lg mb-6">
              <p className="text-lg text-gray-800"><span className="font-bold">데이터셋:</span> {slide.scenario}</p>
            </div>
            <div className="grid grid-cols-2 gap-4 mb-6">
              {slide.results.map((result, i) => (
                <div key={i} className={`p-4 rounded-lg shadow-md ${
                  result.status === 'success' ? 'bg-green-50 border-l-4 border-green-500' :
                  result.status === 'warning' ? 'bg-yellow-50 border-l-4 border-yellow-500' :
                  'bg-gray-50 border-l-4 border-gray-500'
                }`}>
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="font-bold text-lg text-gray-800">{result.metric}</h4>
                      <p className="text-xl font-semibold text-gray-700">{result.value}</p>
                    </div>
                    {result.status === 'success' && <CheckCircle className="w-8 h-8 text-green-500" />}
                    {result.status === 'warning' && <AlertCircle className="w-8 h-8 text-yellow-500" />}
                  </div>
                </div>
              ))}
            </div>
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6 rounded-lg text-center">
              <p className="text-xl font-semibold">{slide.validation}</p>
            </div>
          </div>
        );

      case 'implications':
        return (
          <div className="p-12 h-full bg-gradient-to-br from-green-50 to-blue-50 overflow-y-auto">
            <h2 className="text-3xl font-bold text-gray-800 mb-8">{slide.title}</h2>
            <div className="space-y-6">
              {slide.points.map((point, i) => (
                <div key={i} className="bg-white p-6 rounded-lg shadow-lg border-l-4 border-green-500">
                  <h3 className="text-xl font-bold text-gray-800 mb-2">{point.title}</h3>
                  <p className="text-lg text-gray-700 mb-2">{point.desc}</p>
                  <div className="bg-green-50 p-3 rounded">
                    <p className="text-base text-green-800"><span className="font-semibold">영향:</span> {point.impact}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        );

      case 'future':
        return (
          <div className="p-12 h-full bg-white overflow-y-auto">
            <h2 className="text-3xl font-bold text-gray-800 mb-8">{slide.title}</h2>
            <div className="grid grid-cols-2 gap-6">
              {slide.tracks.map((track, i) => (
                <div key={i} className="bg-gradient-to-br from-purple-50 to-blue-50 p-6 rounded-lg shadow-md">
                  <h3 className="text-xl font-bold text-purple-600 mb-4">{track.category}</h3>
                  <ul className="space-y-2">
                    {track.items.map((item, j) => (
                      <li key={j} className="flex items-start">
                        <span className="text-purple-500 mr-2 mt-1">▸</span>
                        <span className="text-base text-gray-700">{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        );

      case 'conclusion':
        return (
          <div className="p-12 h-full bg-gradient-to-br from-blue-600 to-purple-600 text-white">
            <h2 className="text-4xl font-bold mb-8">{slide.title}</h2>
            <div className="bg-white bg-opacity-20 p-6 rounded-lg mb-6">
              <h3 className="text-2xl font-bold mb-4">주요 기여</h3>
              <ul className="space-y-2">
                {slide.contributions.map((cont, i) => (
                  <li key={i} className="flex items-start">
                    <CheckCircle className="w-6 h-6 mr-3 mt-1 flex-shrink-0" />
                    <span className="text-lg">{cont}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="bg-white bg-opacity-20 p-6 rounded-lg mb-6">
              <h3 className="text-2xl font-bold mb-4">핵심 성과</h3>
              <ul className="space-y-2">
                {slide.keyFindings.map((finding, i) => (
                  <li key={i} className="flex items-start">
                    <span className="text-yellow-300 mr-3 text-xl">★</span>
                    <span className="text-lg">{finding}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="text-center mt-8 p-6 bg-white bg-opacity-30 rounded-lg">
              <p className="text-3xl font-bold mb-2">{slide.message}</p>
              <p className="text-xl italic">{slide.tagline}</p>
            </div>
          </div>
        );

      case 'references':
        return (
          <div className="p-12 h-full bg-gray-50 overflow-y-auto">
            <h2 className="text-3xl font-bold text-gray-800 mb-8">{slide.title}</h2>
            <div className="space-y-4">
              {slide.refs.map((ref, i) => (
                <div key={i} className="bg-white p-4 rounded-lg shadow">
                  <p className="text-base text-gray-700">{ref}</p>
                </div>
              ))}
            </div>
          </div>
        );

      case 'qa':
        return (
          <div className="flex flex-col items-center justify-center h-full bg-gradient-to-br from-blue-600 to-purple-600 text-white p-12">
            <h1 className="text-6xl font-bold mb-12">{slide.title}</h1>
            <div className="bg-white bg-opacity-20 p-8 rounded-lg text-center">
              <p className="text-2xl mb-4">감사합니다</p>
              <p className="text-xl mb-2">{slide.contact.team}</p>
              <p className="text-lg mb-4">{slide.contact.institution}</p>
              <p className="text-xl font-semibold">{slide.contact.email}</p>
            </div>
          </div>
        );

      default:
        return <div className="p-12">Slide type not found</div>;
    }
  };

  return (
    <div className="w-full h-screen bg-gray-900 flex flex-col">
      {/* Main Slide Area */}
      <div className="flex-1 bg-white overflow-hidden">
        {renderSlide(slides[currentSlide])}
      </div>

      {/* Navigation Controls */}
      <div className="bg-gray-800 text-white p-4 flex items-center justify-between">
        <button
          onClick={prevSlide}
          disabled={currentSlide === 0}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 rounded-lg hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition"
        >
          <ChevronLeft className="w-5 h-5" />
          이전
        </button>

        <div className="flex items-center gap-4">
          <span className="text-lg font-semibold">
            {currentSlide + 1} / {slides.length}
          </span>
          <div className="flex gap-1">
            {slides.map((_, i) => (
              <button
                key={i}
                onClick={() => setCurrentSlide(i)}
                className={`w-2 h-2 rounded-full transition ${
                  i === currentSlide ? 'bg-blue-500 w-8' : 'bg-gray-600'
                }`}
              />
            ))}
          </div>
        </div>

        <button
          onClick={nextSlide}
          disabled={currentSlide === slides.length - 1}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 rounded-lg hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition"
        >
          다음
          <ChevronRight className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
};

export default Presentation;