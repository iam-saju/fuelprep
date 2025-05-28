<script>
import QuestionForm from './components/QuestionForm.vue'

export default {
  name: 'App',
  components: {
    QuestionForm
  }
}
</script>

<template>
  <div class="app">
    <header>
      <h1>UPSC Question Generator</h1>
    </header>
    <main>
      <QuestionForm />
    </main>
  </div>
</template>

<style>
.app {
  font-family: Arial, sans-serif;
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

header {
  text-align: center;
  margin-bottom: 40px;
}

h1 {
  color: #2c3e50;
}

main {
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
</style>
