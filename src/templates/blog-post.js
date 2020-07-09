import React from "react"
import { graphql } from "gatsby"
import Layout from "../components/layout"
//import _ from "lodash"
import SEO from "../components/seo"
import Img from "gatsby-image"

import "katex/dist/katex.min.css"
import style from "./blog-post.module.css"

export default function BlogPost({ data }) {
  const post = data.markdownRemark
  return (
    <Layout>
      <SEO
        title={post.frontmatter.title}
        description={post.excerpt}
        image="/jumpML.svg"
        pathname={post.fields.slug}
        // Boolean indicating whether this is an article:
        article
      />
      <article className={style.blogpost}>
        {post.frontmatter.featimg && (
          <figure className={style.featimg}>
            <Img
              fluid={post.frontmatter.featimg.childImageSharp.fluid}
              alt={post.frontmatter.title}
            />
          </figure>
        )}
        <h1 className={style.blogpost__title}>{post.frontmatter.title}</h1>
        <div className={style.blogpost__readtime}>
          ~ {post.fields.readingTime.text}
        </div>
        <div className={style.blogpost__meta}>
          by {post.frontmatter.author}. Published{" "}
          {new Date(post.frontmatter.date).toLocaleDateString("en-US", {
            month: "long",
            day: "numeric",
            year: "numeric",
          })}{" "}
        </div>
        <div className={style.blogpost__tax}>
          Filed under:{" "}
          {post.frontmatter.subject.map((subject, index) => [
            index > 0 && ", ",
            //          <Link key={index} to={`/subjects/${_.kebabCase(subject)}`}>
            subject,
            //          </Link>,
          ])}
        </div>

        <div
          className={style.blogpost__content}
          dangerouslySetInnerHTML={{ __html: post.html }}
        />
      </article>
    </Layout>
  )
}

export const query = graphql`
  query($slug: String!) {
    markdownRemark(fields: { slug: { eq: $slug } }) {
      html
      excerpt(pruneLength: 160)
      frontmatter {
        title
        date
        subject
        author
        featimg {
          childImageSharp {
            fluid(maxWidth: 1360) {
              ...GatsbyImageSharpFluid
            }
          }
        }
      }
      fields {
        slug
        readingTime {
          text
        }
      }
    }
    site {
      siteMetadata {
        title
      }
    }
  }
`
