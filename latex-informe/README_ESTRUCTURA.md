# Estructura Modular de la Tesina

## Archivos Creados

La tesina ha sido dividida en los siguientes archivos:

### Capítulos Principales
- `introduccion.tex` - Capítulo 1: Introducción
- `marco_teorico.tex` - Capítulo 2: Marco Teórico (incluye las subsecciones)
  - `marco_teorico-conceptos_basicos.tex`
  - `marco_teorico-clasificadores.tex`
  - `marco_teorico-redes_neuronales_artificiales.tex`
- `formulacion_del_problema.tex` - Capítulo 3: Formulación del problema
- `trabajos_relacionados_y_estado_del_arte.tex` - Capítulo 4: Trabajos relacionados
- `experimentos.tex` - Capítulo 5: Experimentos

### Apéndices
- `construccion_del_conjunto_de_datos.tex` - Apéndice A
- `trabajo_futuro.tex` - Apéndice B

### Archivo Principal
- `informe.tex` - Archivo principal que incluye todos los capítulos

## Compilación

### Compilación Completa

Para compilar el documento completo:

```bash
make compile
```

o

```bash
make latexmk
```

### Compilación Parcial (Rápida)

Para compilar solo capítulos específicos (mucho más rápido):

```bash
# Compilar solo la introducción
make intro

# Compilar solo el marco teórico
make teoria

# Compilar solo la formulación del problema
make formulacion

# Compilar solo trabajos relacionados
make trabajos

# Compilar solo experimentos
make exp

# Compilar solo la construcción del conjunto de datos
make construccion

# Compilar solo el trabajo futuro
make futuro
```

### Compilación Personalizada

También puedes compilar múltiples capítulos específicos:

```bash
make fast ONLY='introduccion marco_teorico'
```

Capítulos disponibles:
- `introduccion`
- `marco_teorico`
- `formulacion_del_problema`
- `trabajos_relacionados_y_estado_del_arte`
- `experimentos`
- `construccion_del_conjunto_de_datos`
- `trabajo_futuro`

## Ventajas de la Estructura Modular

1. **Compilación más rápida**: Compila solo las partes que estás editando
2. **Mejor organización**: Cada capítulo en su propio archivo
3. **Más fácil de mantener**: Cambios aislados por capítulo
4. **Trabajo en equipo**: Múltiples personas pueden trabajar en diferentes capítulos
5. **Control de versiones**: Git puede rastrear mejor los cambios

## Utilidades del Makefile

```bash
make clean     # Limpia archivos auxiliares
make cleanall  # Limpia todo incluyendo el PDF
make view      # Abre el PDF
make show      # Compila y abre el PDF
make help      # Muestra ayuda
```

## Notas Importantes

- Las imágenes ahora están en el directorio `imgs/`
- El archivo de respaldo original está en `main_backup.tex`
- La compilación parcial usa `\includeonly` de LaTeX para mayor velocidad
- Las referencias cruzadas entre capítulos funcionan normalmente

## Ejemplo de Flujo de Trabajo

1. Editar un capítulo, por ejemplo `introduccion.tex`
2. Compilar solo ese capítulo: `make intro`
3. Ver el resultado: `make view`
4. Cuando termines, compilar todo: `make compile`

## Solución de Problemas

Si las referencias cruzadas no funcionan correctamente:
```bash
make cleanall
make compile
```

Esto forzará una recompilación completa que resolverá las referencias.

