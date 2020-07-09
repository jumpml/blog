import React from "react"
import { Helmet } from "react-helmet"
import { useStaticQuery, graphql } from "gatsby"

const SEO = ({ title, description, image, path, article }) => {
  const { site } = useStaticQuery(
    graphql`
      query {
        site {
          siteMetadata {
            title
            description
            siteUrl
            logo
            social {
              twitter
            }
          }
        }
      }
    `
  )
  const twitterUsername = site.siteMetadata.social.twitter
  const siteUrl = site.siteMetadata.siteUrl

  const seo = {
    defaultTitle: site.siteMetadata.title,
    title: title || site.siteMetadata.title,
    description: description || site.siteMetadata.description,
    image: `${siteUrl}${image || site.siteMetadata.logo}`,
    url: `${siteUrl}${path || "/"}`,
  }

  return (
    <>
      <Helmet title={seo.title}>
        <meta charSet="utf-8" />
        <meta name="description" content={seo.description} />
        <meta name="image" content={seo.image} />
        {seo.url && <meta property="og:url" content={seo.url} />}
        {seo.url && <meta rel="canonical" content={seo.url} />}
        {(article ? true : null) && (
          <meta property="og:type" content="article" />
        )}
        {seo.title && <meta property="og:title" content={seo.title} />}
        <meta property="og:site_name" content={seo.defaultTitle} />
        {seo.description && (
          <meta property="og:description" content={seo.description} />
        )}
        {seo.image && <meta property="og:image" content={seo.image} />}
        <meta name="twitter:card" content="summary_large_image" />
        {twitterUsername && (
          <meta name="twitter:creator" content={twitterUsername} />
        )}
        {seo.title && <meta name="twitter:title" content={seo.title} />}
        {seo.description && (
          <meta name="twitter:description" content={seo.description} />
        )}
        {seo.image && <meta name="twitter:image" content={seo.image} />}
        <link rel="canonical" href={seo.url} />
      </Helmet>
    </>
  )
}

export default SEO
