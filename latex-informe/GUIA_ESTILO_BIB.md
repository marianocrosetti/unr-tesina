# Guía de Estilo para la Bibliografía

## Formato Estándar Aplicado

### Estructura General
```latex
\bibitem[Autor Año]{etiqueta}
Autor(es). (Año).
\textit{Título del trabajo}.
Publicación/Venue/Detalles.
URL (si aplica)
DOI (si aplica)
```

### Reglas de Formato

1. **Autor y Año en el \bibitem**:
   - Formato: `[Autor Año]` o `[Autor et al. Año]`
   - Ejemplo: `\bibitem[Smith 2015]{smith2015}`
   - Si hay múltiples trabajos del mismo autor/año: agregar a, b, c...

2. **Nombres de Autores**:
   - Primer autor: Apellido, Iniciales.
   - Autores adicionales: and Apellido, I.
   - Más de 3 autores: usar "et al."
   - Ejemplo: `Smith, J., Doe, A., and Johnson, M.`

3. **Año**:
   - Entre paréntesis después de los autores
   - Ejemplo: `(2015).`

4. **Título**:
   - En cursiva usando `\textit{}`
   - Mayúscula inicial y nombres propios
   - Terminar con punto
   - Ejemplo: `\textit{Deep Learning for Computer Vision}.`

5. **Publicación/Venue**:
   - Sin cursiva
   - Incluir: journal, conferencia, editorial, páginas
   - Ejemplo: `In Proceedings of NeurIPS, pp. 1234-1245.`

6. **URLs**:
   - Usar `\url{}`
   - En línea separada
   - Ejemplo: `\url{https://arxiv.org/abs/1234.5678}`

7. **DOI**:
   - Formato: `doi:10.XXXX/YYYY`
   - Después del URL si existe

8. **Saltos de línea**:
   - NO usar `\\` al final de las líneas
   - Usar punto (`.`) al final de cada campo

## Cambios Aplicados

### Homogeneizados (líneas 1-460 aproximadamente):
- ✅ Eliminado bibitem vacío
- ✅ Estandarizado formato de autores
- ✅ Estandarizado uso de años
- ✅ Estandarizado títulos (todos en cursiva)
- ✅ Eliminados saltos de línea innecesarios (`\\`)
- ✅ Estandarizado formato de URLs
- ✅ Corregidas inconsistencias en puntuación
- ✅ Mejorada legibilidad general

### Pendientes (líneas 460-785):
- Secciones de competencias (SigComp)
- Trabajos de verificación de firmas
- Trabajos de redes siamesas y face recognition
- Trabajos más recientes (2014-2019)

## Ejemplo de Transformación

### Antes:
```latex
\bibitem[Hao Li et. al. 2018]{visualizing-landscapes}
Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein 2018
\textit{ Visualizing the Loss Landscape of Neural Nets } \\
University of Maryland, College Park United States Naval Academy Cornell University
```

### Después:
```latex
\bibitem[Li et al. 2018]{visualizing-landscapes}
Li, H., Xu, Z., Taylor, G., Studer, C., and Goldstein, T. (2018).
\textit{Visualizing the Loss Landscape of Neural Nets}.
University of Maryland, College Park United States Naval Academy Cornell University.
```

## Verificación

Para verificar los cambios:
```bash
make bib        # Compilar bibliografía
make showbib    # Ver resultado
```

