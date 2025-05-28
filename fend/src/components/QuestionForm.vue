<template>
  <div class="question-form">
    <h2>Select a Topic</h2>
    
    <div class="topic-cards">
      <div 
        v-for="(topic, index) in topics" 
        :key="`topic-${index}-${topicKeys[index]}`"
        class="topic-card"
        :class="{ 
          'is-flipped': selectedTopic === topic,
          'is-processing': processingCards.includes(index),
          'slide-out': slidingOutCards.includes(index)
        }"
        @click="selectTopic(topic, index)"
      >
        <div class="card-inner">
          <div class="card-front">
            <div class="topic-text">{{ topic }}</div>
          </div>
          <div class="card-back">
            <div class="loading-spinner" v-if="loading && selectedTopic === topic"></div>
            <div v-else class="back-content">
              <div class="icon">🎯</div>
              <div>Generate Questions</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div v-if="error" class="error">
      {{ error }}
    </div>

    <div v-if="questions.length > 0" class="response">
      <h3>Generated Questions:</h3>
      <div class="questions-container">
        <div class="question-section">
          <h4>Prelims (MCQ) Questions:</h4>
          <div v-for="(question, index) in prelimsQuestions" :key="'prelims-'+index" class="question-content">
            {{ question }}
          </div>
        </div>
        <div class="question-section">
          <h4>Mains (Descriptive) Questions:</h4>
          <div v-for="(question, index) in mainsQuestions" :key="'mains-'+index" class="question-content">
            {{ question }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'QuestionForm',
  data() {
    return {
      topics: [],
      selectedTopic: '',
      questions: [],
      loading: false,
      error: null,
      processingCards: [],
      slidingOutCards: [],
      topicKeys: [], // To force re-render when topics change
      usedTopics: new Set(), // Track used topics to avoid repetition
      allAvailableTopics: [] // Store all available topics
    }
  },
  computed: {
    prelimsQuestions() {
      return this.questions.filter(q => 
        q.toLowerCase().includes('prelims') || 
        q.toLowerCase().includes('mcq') ||
        q.includes('(A)') || q.includes('(B)') || q.includes('(C)') || q.includes('(D)')
      ).slice(0, 5)
    },
    mainsQuestions() {
      return this.questions.filter(q => 
        q.toLowerCase().includes('mains') || 
        q.toLowerCase().includes('descriptive') ||
        (!q.includes('(A)') && !q.includes('(B)') && !q.includes('(C)') && !q.includes('(D)'))
      ).slice(0, 5)
    }
  },
  async created() {
    await this.fetchTopics()
  },
  methods: {
    async fetchTopics() {
      try {
        const response = await fetch('/api/topics')
        if (!response.ok) {
          throw new Error('Failed to fetch topics')
        }
        const data = await response.json()
        this.allAvailableTopics = [...data.topics]
        this.topics = data.topics.slice(0, 6) // Show first 6 topics
        this.topicKeys = this.topics.map((_, index) => index)
      } catch (err) {
        this.error = 'Error: ' + err.message
        // Fallback topics for demo
        this.allAvailableTopics = [
          'Indian Constitution',
          'Modern History', 
          'Geography',
          'Economy',
          'Polity',
          'Environment',
          'Ancient History',
          'Medieval History',
          'Art & Culture',
          'Science & Technology',
          'International Relations',
          'Internal Security',
          'Disaster Management',
          'Social Issues',
          'Ethics',
          'Agriculture',
          'Infrastructure',
          'Energy',
          'Climate Change',
          'Biodiversity'
        ]
        this.topics = this.allAvailableTopics.slice(0, 6)
        this.topicKeys = this.topics.map((_, index) => index)
      }
    },

    async selectTopic(topic, index) {
      if (this.loading || this.processingCards.includes(index)) return
      
      // Immediately flip the card and show loading
      this.selectedTopic = topic
      this.processingCards.push(index)
      this.questions = []
      this.error = null
      this.loading = true

      // Force a small delay to ensure the flip animation is visible
      await new Promise(resolve => setTimeout(resolve, 100))

      try {
        // Simulate API call for demo - replace with your actual API
        await new Promise(resolve => setTimeout(resolve, 2000))
        
        // Mock response for demo - replace with actual API call
        const mockQuestions = [
          `Prelims Question 1: Which article of the Indian Constitution deals with ${topic}?\n(A) Article 19\n(B) Article 21\n(C) Article 32\n(D) Article 356`,
          `Prelims Question 2: The concept related to ${topic} was first introduced in which year?\n(A) 1947\n(B) 1950\n(C) 1976\n(D) 1991`,
          `Prelims Question 3: Which committee recommended reforms in ${topic}?\n(A) Sarkaria Commission\n(B) Rajmannar Committee\n(C) Shah Commission\n(D) Thakkar Commission`,
          `Prelims Question 4: The ${topic} is primarily governed by which ministry?\n(A) Home Ministry\n(B) Finance Ministry\n(C) External Affairs\n(D) Law Ministry`,
          `Prelims Question 5: Which of the following best describes ${topic}?\n(A) Constitutional provision\n(B) Statutory requirement\n(C) Administrative guideline\n(D) Judicial precedent`,
          `Mains Question 1: Critically analyze the role of ${topic} in modern Indian governance. Discuss its evolution and current challenges.`,
          `Mains Question 2: Examine the constitutional and legal framework governing ${topic}. How has judicial interpretation shaped its implementation?`,
          `Mains Question 3: Discuss the impact of ${topic} on federalism in India. Analyze both positive and negative aspects.`,
          `Mains Question 4: Evaluate the recent reforms in ${topic}. What further changes are needed for effective implementation?`,
          `Mains Question 5: Compare the Indian approach to ${topic} with international best practices. What lessons can be learned?`
        ]

        /*
        // Uncomment this for actual API call
        const response = await fetch('/api/generate_question', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ 
            prompt: `Generate 10 UPSC-style questions about ${topic}. Include 5 Prelims (MCQ) questions with options (A), (B), (C), (D) and 5 Mains (descriptive) questions. Format each question clearly and separate them with double line breaks.`
          })
        })

        if (!response.ok) {
          throw new Error('Failed to generate questions')
        }

        const data = await response.json()
        this.questions = data.question.split('\n\n').filter(q => q.trim())
        */
        
        this.questions = mockQuestions
        
        // Mark topic as used
        this.usedTopics.add(topic)
        
        // Wait a moment to show the completed state
        await new Promise(resolve => setTimeout(resolve, 500))
        
        // Start card replacement animation
        await this.replaceCard(index)
        
      } catch (err) {
        this.error = 'Error: ' + err.message
      } finally {
        this.loading = false
        this.selectedTopic = ''
        // Remove from processing cards
        this.processingCards = this.processingCards.filter(i => i !== index)
      }
    },

    async replaceCard(index) {
      // Add slide-out animation
      this.slidingOutCards.push(index)
      
      // Wait for slide-out animation
      await new Promise(resolve => setTimeout(resolve, 600))
      
      // Get a new topic
      const newTopic = this.getNewTopic()
      
      if (newTopic) {
        // Replace the topic
        this.$set(this.topics, index, newTopic)
        // Update key to force re-render
        this.$set(this.topicKeys, index, Date.now() + Math.random())
      }
      
      // Remove slide-out class after a short delay
      setTimeout(() => {
        this.slidingOutCards = this.slidingOutCards.filter(i => i !== index)
      }, 100)
    },

    getNewTopic() {
      // Find topics that haven't been used yet
      const availableTopics = this.allAvailableTopics.filter(topic => 
        !this.usedTopics.has(topic) && !this.topics.includes(topic)
      )
      
      if (availableTopics.length > 0) {
        return availableTopics[Math.floor(Math.random() * availableTopics.length)]
      }
      
      // If all topics are used, reset and start over (optional)
      if (this.usedTopics.size >= this.allAvailableTopics.length) {
        this.usedTopics.clear()
        return this.allAvailableTopics[Math.floor(Math.random() * this.allAvailableTopics.length)]
      }
      
      return null
    }
  }
}
</script>

<style scoped>
.question-form {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.topic-cards {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 25px;
  margin-bottom: 40px;
  perspective: 1000px;
}

.topic-card {
  height: 160px;
  cursor: pointer;
  position: relative;
  transform-style: preserve-3d;
  transition: all 0.6s cubic-bezier(0.4, 0.0, 0.2, 1);
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
}

.topic-card:hover {
  transform: translateY(-5px) scale(1.02);
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.18);
}

.topic-card.is-flipped {
  transform: rotateY(180deg);
}

.topic-card.is-processing {
  transform: rotateY(180deg) scale(1.05);
  box-shadow: 0 16px 40px rgba(76, 175, 80, 0.3);
}

.topic-card.slide-out {
  transform: translateX(-100vw) rotateZ(-15deg);
  opacity: 0;
  transition: all 0.6s cubic-bezier(0.6, 0.0, 0.8, 1);
}

.card-inner {
  position: relative;
  width: 100%;
  height: 100%;
  text-align: center;
  transition: transform 0.6s;
  transform-style: preserve-3d;
}

.card-front, .card-back {
  position: absolute;
  width: 100%;
  height: 100%;
  -webkit-backface-visibility: hidden;
  backface-visibility: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  border-radius: 12px;
  box-sizing: border-box;
  border: 3px solid transparent;
}

.card-front {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: 3px solid rgba(255, 255, 255, 0.2);
  position: relative;
  overflow: hidden;
}

.card-front::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(45deg, transparent 30%, rgba(255, 255, 255, 0.1) 50%, transparent 70%);
  transform: translateX(-100%);
  transition: transform 0.6s;
}

.topic-card:hover .card-front::before {
  transform: translateX(100%);
}

.topic-text {
  font-weight: bold;
  font-size: 1.2em;
  text-align: center;
  word-wrap: break-word;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
  z-index: 1;
  position: relative;
}

.card-back {
  background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
  color: white;
  transform: rotateY(180deg);
  font-size: 1.1em;
  border: 3px solid rgba(255, 255, 255, 0.2);
  flex-direction: column;
  gap: 10px;
}

.back-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}

.icon {
  font-size: 2em;
  animation: bounce 2s infinite;
}

@keyframes bounce {
  0%, 20%, 50%, 80%, 100% {
    transform: translateY(0);
  }
  40% {
    transform: translateY(-10px);
  }
  60% {
    transform: translateY(-5px);
  }
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid rgba(255, 255, 255, 0.3);
  border-top: 4px solid #ffffff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.error {
  color: #d32f2f;
  margin: 20px 0;
  padding: 15px;
  background-color: #ffebee;
  border-radius: 8px;
  border-left: 4px solid #d32f2f;
  font-weight: 500;
}

.response {
  margin-top: 40px;
  animation: fadeInUp 0.6s ease-out;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.response h3 {
  color: #2c3e50;
  margin-bottom: 25px;
  font-size: 1.5em;
  text-align: center;
}

.questions-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 30px;
}

.question-section {
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  padding: 25px;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.question-section h4 {
  color: #2c3e50;
  margin-bottom: 20px;
  padding-bottom: 15px;
  border-bottom: 3px solid #667eea;
  font-size: 1.2em;
  position: relative;
}

.question-section h4::after {
  content: '';
  position: absolute;
  bottom: -3px;
  left: 0;
  width: 50px;
  height: 3px;
  background: linear-gradient(90deg, #667eea, #764ba2);
}

.question-content {
  margin-bottom: 20px;
  padding: 18px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  white-space: pre-wrap;
  line-height: 1.6;
  border-left: 4px solid #667eea;
  transition: transform 0.2s, box-shadow 0.2s;
}

.question-content:hover {
  transform: translateX(5px);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
}

@media (max-width: 768px) {
  .questions-container {
    grid-template-columns: 1fr;
  }
  
  .topic-cards {
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 15px;
  }
  
  .topic-card {
    height: 140px;
  }
  
  .topic-text {
    font-size: 1em;
  }
}

@media (max-width: 480px) {
  .topic-cards {
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  }
  
  .card-front, .card-back {
    padding: 15px;
  }
}
</style>