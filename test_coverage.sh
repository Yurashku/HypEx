#!/bin/bash

# Скрипт для запуска тестов с генерацией покрытия
# Использование: ./test_coverage.sh

export PYTHONPATH=$PYTHONPATH:.
export PYTHONWARNINGS="ignore"

echo "🚀 Запуск тестов с измерением покрытия..."

# Запуск тестов с покрытием
python3 -m pytest tests/ \
  --cov=hypex \
  --cov-report=html:htmlcov \
  --cov-report=term-missing \
  --cov-report=xml:coverage.xml \
  --cov-branch \
  --cov-fail-under=80 \
  -v

echo ""
echo "📊 Покрытие измерено!"
echo "📁 HTML-отчет сгенерирован в: htmlcov/index.html"
echo "📁 XML-отчет сгенерирован в: coverage.xml"
echo ""
echo "🌐 Для просмотра HTML-отчета:"
echo "   firefox htmlcov/index.html"
echo "   или"
echo "   python3 -m http.server 8000 -d htmlcov"